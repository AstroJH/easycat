import sys
from os import path
from typing import Callable, Iterable, Any
import time
import signal

from pandas import DataFrame, Series
import numpy as np
from numpy import recarray

from concurrent.futures import ThreadPoolExecutor, Future
from progressbar import ProgressBar, Percentage, Bar, Timer, Counter

RecordIterable = Iterable[tuple[str,bool]]
Record = tuple[str,bool]


def start(logpath:str, catalog:DataFrame,
          handler:Callable[[str,dict],bool],
          exception_handler:Callable[[BaseException],Any]=(lambda e: print(e)),
          n_workers:int=10):
    
    download_records = load_logfile(logpath, catalog)

    td_pool = ThreadPoolExecutor(max_workers=n_workers)

    signal.signal(signal.SIGINT, signal.SIG_IGN) # <========================== ignore Ctrl+C

    futures:list[Future] = []
    for record in download_records:
        obj_id, is_completed = record
        if is_completed: continue

        row:Series = catalog.loc[obj_id] # BUG if ... not found?

        f = td_pool.submit(task4record, record, handler, row.to_dict())
        futures.append(f)

    shutdown = shutdown_builder(td_pool, logpath, download_records, futures)
    signal.signal(signal.SIGINT, shutdown) # <================================ handle Ctrl+C
    
    show_progress_bar(download_records) # <= an infinite loop

    # clean resources and save download logfile
    td_pool.shutdown(wait=True)
    np.save(file=logpath, arr=download_records)

    check_exceptions(futures, exception_handler)


def task4record(record:Record, handler:Callable[[str,dict],bool], param:dict):
    obj_id, is_completed = record
    if is_completed:
        return

    try:
        is_successful = handler(obj_id, param)
        record[1] = is_successful
    except Exception as e:
        record[1] = False
        raise e


def show_progress_bar(download_records:RecordIterable):
    n_record = len(download_records)

    widgets = [
        Percentage(), " ",
        Bar(), " ",
        Timer(format="%s"), " ",
        Counter(), f"/{n_record}"
    ]
    
    pbar = ProgressBar(widgets=widgets, maxval=n_record).start()
    while True:
        # BUG 如果存在未下载成功的数据
        # n_completed >= n_record 将始终不会满足，即此处会陷入死循环
        # 必须检查线程是否已经跑完
        # 目前只能依赖用户使用 Ctrl+C 来终止任务

        n_completed = np.sum(download_records["is_completed"])
        
        pbar.update(n_completed)

        if n_completed >= n_record:
            pbar.finish()
            break
        time.sleep(5)


def check_exceptions(futures:list[Future], handler:Callable[[BaseException],Any]=(lambda e: print(e))):
    for f in futures:
        if f.cancelled():
            continue

        e = f.exception()
        if e:
            handler(e)


def shutdown_builder(td_pool:ThreadPoolExecutor,
                     logpath:str, download_record,
                     futures:list[Future]):
    def shutdown(s, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN) # Don't fire Ctrl+C repeatedly!

        print("\nStopping the task, please be patient...")
        td_pool.shutdown(wait=True, cancel_futures=True)
        np.save(file=logpath, arr=download_record)

        n_completed = np.sum(download_record["is_completed"])
        print(f"Download Process: {n_completed}/{len(download_record)}")

        check_exceptions(futures)
        sys.exit(0)
    
    return shutdown


def load_logfile(logpath:str, catalog:DataFrame) -> RecordIterable:
    if not path.isfile(logpath):
        print("Creating log file... ", end="")
        obj_ids:list[str] = list(catalog.index)

        download_record = recarray((len(obj_ids),), dtype=[("obj_id", "<U32"), ("is_completed", "u1")])
        for i in range(0, len(obj_ids)):
            download_record[i] = (obj_ids[i], False)

        np.save(file=logpath, arr=download_record)
        print("Completed.")
    else:
        download_record = np.load(logpath)
    
    return download_record

from . import download
from . import subcat2d
__all__ = ["start", "download", "subcat2d"]