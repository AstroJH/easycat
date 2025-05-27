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

class Record:
    def __init__(self):
        self.completed = set()
        self.failed = set()
        self.results = {}
    
    def reset_failed(self):
        self.failed = set()


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

        last_save = time.time()

        # default_handler = signal.SIGINT
        # signal.signal(signal.SIGINT, signal.SIG_IGN) # <========================== ignore Ctrl+C

        # Submit tasks
        # future -> idx -> row
        futures:dict[Future, Any] = {}
        for idx, row in catalog.iterrows():
            if (idx in record["completed"]) or (idx in record["failed"]): continue

            future = executor.submit(task4record, idx, task, row.to_dict())
            futures[future] = idx

        # shutdown = self.shutdown_builder(futures)
        # signal.signal(signal.SIGINT, shutdown) # <================================ handle Ctrl+C

        n_cancel = 0
        try:
            n_total = len(catalog)
            n_count = len(record["completed"])+len(record["failed"])
            
            pbar = init_pbar(n_total)
            pbar.update(n_count)

            for future in as_completed(futures):
                n_count += 1
                idx = futures[future]
                handle_future(idx, record, future)

                pbar.update(n_count)

                # periodic checkpoint save
                if (time.time() - last_save) > self.checkpoint_interval:
                    self.save_checkpoint()
                    last_save = time.time()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            print("\nStopping the task, please be patient...")

            remain_futures = []

            for future in futures.keys():
                if future.cancel(): n_cancel += 1
                elif future.running():
                    remain_futures.append(future)
            
            n_running = len(remain_futures)

            print(f"{n_cancel} tasks have been cancelled")
            print(f"{n_running} tasks are running")

            for future in as_completed(remain_futures):
                idx = futures[future]
                handle_future(idx, record, future)
            print("Stopped.")
        finally:
            # clean resources and save logfile
            executor.shutdown(wait=True)
            self.save_checkpoint()
            check_exceptions(futures, self.exception_handler)

            n_completed = len(record.get("completed"))
            n_failed = len(record.get("failed"))
            total = len(catalog)

            print("\nSummary")
            print("=============")
            print("Total:", total)
            print("Completed:", n_completed)
            print("Failed:", n_failed)
            print("Cancelled:", n_cancel)

            if n_cancel != total - n_failed - n_completed:
                print("A panic! len(catalog) != n_failed + n_completed + n_cancelled")
        
        return record
    
    def save_checkpoint(self):
        if self.checkpoint is None:
            return

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
            # print(self.executor.)
            self.executor.shutdown(wait=True, cancel_futures=True)
            
            self.save_checkpoint()

            check_exceptions(futures)
        
        return shutdown


def handle_future(idx, record, future:Future):
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

