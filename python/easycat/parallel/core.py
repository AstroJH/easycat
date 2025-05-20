import sys
import os
from os import path
from typing import Callable, Any, Literal, Optional
import time
import signal
import pickle
from tempfile import NamedTemporaryFile

from pandas import DataFrame

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from concurrent.futures._base import Executor
from progressbar import ProgressBar, Percentage, Bar, Timer, Counter


class TaskDispatcher:
    def __init__(
        self, catalog:DataFrame, task:Callable[[str,dict],tuple], *,
        mode: Literal["thread", "process"]="thread",
        n_workers:int=4,
        checkpoint:Optional[str]=None,
        checkpoint_interval=10,
        rehandle_failed:bool=False,
        exception_handler:Callable[[BaseException],Any]=(lambda e: print(e)),
    ): 
        if mode not in ("thread", "process"):
            raise ValueError("Invalid mode. Choose 'thread' or 'process'")
        
        self.catalog = catalog
        self.task = task
        self.mode = mode
        self.n_workers = n_workers
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.rehandle_failed = rehandle_failed
        self.exception_handler = exception_handler

        self.load_checkpoint()
        self.executor:Executor = (
            ThreadPoolExecutor if self.mode == "thread" else ProcessPoolExecutor
        )(max_workers=self.n_workers)

        if rehandle_failed:
            self.record["failed"] = set()
        

    def dispatch(self):
        catalog = self.catalog
        record = self.record
        task = self.task
        executor = self.executor
        checkpoint = self.checkpoint

        last_save = time.time()

        signal.signal(signal.SIGINT, signal.SIG_IGN) # <========================== ignore Ctrl+C

        # Submit tasks
        # future -> idx -> row
        futures = {}
        for idx, row in catalog.iterrows():
            if (idx in record["completed"]) or (idx in record["failed"]): continue

            future = executor.submit(task4record, idx, task, row.to_dict())
            futures[future] = idx

        shutdown = self.shutdown_builder(futures)
        signal.signal(signal.SIGINT, shutdown) # <================================ handle Ctrl+C

        n_total = len(catalog)
        n_count = len(record["completed"])+len(record["failed"])
        
        pbar = init_pbar(n_total)
        pbar.update(n_count)

        for future in as_completed(futures):
            n_count += 1
            idx = futures[future]
            try:
                is_successful, data = future.result()

                if is_successful:
                    record["completed"].add(idx)
                    record["results"][idx] = data
                else:
                    record["failed"].add(idx)
                    record["results"][idx] = None

            except Exception as e:
                record["failed"].add(idx)
                record["results"][idx] = None
            
            pbar.update(n_count)

            # Periodic checkpoint save
            if checkpoint and (time.time() - last_save) > self.checkpoint_interval:
                self.save_checkpoint()
                last_save = time.time()

        # clean resources and save logfile
        executor.shutdown(wait=True)

        if checkpoint: self.save_checkpoint()

        check_exceptions(futures, self.exception_handler)
        return record
    
    def save_checkpoint(self):
        record = self.record
        filepath = self.checkpoint

        # Safely save checkpoint file
        with NamedTemporaryFile("wb", delete=False) as f:
            pickle.dump(record, f)
            tempname = f.name
        os.replace(tempname, filepath)


    def load_checkpoint(self) -> dict:
        filepath = self.checkpoint

        record = {
            "completed": set(),
            "failed": set(),
            "results": {}
        }

        if filepath is None:
            self.record = record
            return
        
        if not path.exists(filepath):
            print(f"Creating log file [{filepath}] ... ", end="")
            self.record = record
            self.save_checkpoint()
            print("Completed.")
            return
        
        if path.isfile(filepath):
            with open(filepath, "rb") as f:
                record = pickle.load(f)
            
            completed = record["completed"]
            print(f"Checkpoint file detected. Resumed {len(completed)} records.")

            self.record = record
            return
        
        print(f"[WARNING] {filepath} is not a file.")
        self.record = record
    
    def shutdown_builder(self, futures:list[Future]):
        def shutdown(s, frame):
            signal.signal(signal.SIGINT, signal.SIG_IGN) # Don't fire Ctrl+C repeatedly!

            print("\nStopping the task, please be patient...")
            self.executor.shutdown(wait=True, cancel_futures=True)
            
            self.save_checkpoint()

            check_exceptions(futures)
            sys.exit(0)
        
        return shutdown
    


def task4record(idx, handler:Callable[[str,dict],dict], param:dict):
    is_successful, data = handler(idx, param)
    return is_successful, data


def check_exceptions(futures:list[Future], handler:Callable[[BaseException],Any]=(lambda e: print(e))):
    for f in futures:
        if f.cancelled():
            continue

        e = f.exception()
        if e:
            handler(e)


def init_pbar(n_total):
    widgets = [
            Percentage(), " ",
            Bar(), " ",
            Timer(format="%s"), " ",
            Counter(), f"/{n_total}"
        ]
    
    pbar = ProgressBar(widgets=widgets, maxval=n_total).start()
    return pbar

