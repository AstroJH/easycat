import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from progressbar import ProgressBar, Percentage, Bar, Timer, Counter
import signal
import os
from typing import Callable, Optional, Literal
import time
import sys

class SimpleInterrupt:
    def __init__(self, checkpoint_data:dict, executor):
        self.interrupted = False
        self.checkpoint_data = checkpoint_data
        self.original_sigint = signal.getsignal(signal.SIGINT)
        self.executor = executor
        
    def __enter__(self):
        """Register signal handler"""
        signal.signal(signal.SIGINT, self._handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original signal handler"""
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def _handler(self, signum, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN) # Don't fire Ctrl+C repeatedly!

        self.interrupted = True
        print("\nStopping the task, please be patient...")
        
        self.executor.shutdown(wait=True, cancel_futures=True)
        np.save(file=logpath, arr=download_record)

        n_completed = np.sum(download_record["is_completed"])
        print(f"Download Process: {n_completed}/{len(download_record)}")

        check_exceptions(futures)
        sys.exit(0)




def dispatch(
    df: pd.DataFrame,
    task: Callable,
    mode: Literal["thread", "process"],
    max_workers: Optional[int] = None,
    checkpoint: Optional[str] = None,
    checkpoint_interval: int = 10
) -> pd.DataFrame:
    """
    Parallel processing with interruption and resume support
    
    Args:
        df: Input DataFrame to process
        task: Processing function that accepts (index, row) tuple
        mode: Execution mode ("thread"/"process")
        max_workers: Maximum concurrent workers
        checkpoint: Path for progress checkpoint file
        checkpoint_interval: Auto-save interval in seconds
    """
    
    # Validate parameters
    if mode not in ("thread", "process"):
        raise ValueError("Invalid mode. Choose 'thread' or 'process'")
        
    # Initialize results container
    results = {}
    checkpoint_data = {
        "processed": set(),
        "results": {}
    }
    
    # Load existing checkpoint
    if checkpoint and os.path.exists(checkpoint):
        import pickle
        with open(checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint file detected. Resumed {len(checkpoint_data['processed'])} records")

    # Create executor
    Executor = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
    max_workers = max_workers or (os.cpu_count() or 4)
    
    with SimpleInterrupt() as interrupt, \
        Executor(max_workers=max_workers) as executor:
        
        # Submit tasks
        # future -> index -> row
        futures = {}
        for index, row in df.iterrows():
            if index not in checkpoint_data["processed"]:
                future = executor.submit(task, (index, row))
                futures[future] = index
                
        # Progress bar configuration
        total = len(futures)
        last_save = time.time()

        show_progress_bar(total, checkpoint_data) # <= an infinite loop
        
        # with tqdm(total=total) as pbar:
        #     set_desc4tqdm(pbar, mode, total)
        #     try:
        #         for future in as_completed(futures):
        #             if interrupt.interrupted:
        #                 break
                        
        #             # Process result
        #             index = futures[future]
        #             try:
        #                 result = future.result()
        #                 results[index] = result
        #                 checkpoint_data["processed"].add(index)
        #                 checkpoint_data["results"][index] = result
        #             except Exception as e:
        #                 print(f"\nTask {index} failed: {str(e)[:200]}")
                        
        #             # Update progress
        #             pbar.update(1)
        #             set_desc4tqdm(pbar, mode, len(futures)-pbar.n)
                    
        #             # Periodic checkpoint save
        #             if checkpoint and (time.time() - last_save) > checkpoint_interval:
        #                 _save_checkpoint(checkpoint, checkpoint_data)
        #                 last_save = time.time()
                        
        #     except KeyboardInterrupt:
        #         if not interrupt.interrupted:
        #             raise  # Handle non-signal interrupts
                    
        # # Final checkpoint save
        # if checkpoint:
        #     _save_checkpoint(checkpoint, checkpoint_data)
            
    # Merge results
    final_results = {**checkpoint_data["results"], **results}
    return pd.DataFrame.from_dict(final_results, orient='index').reindex(df.index)


def _save_checkpoint(path: str, data: dict):
    """Safely save checkpoint file"""

    import pickle
    from tempfile import NamedTemporaryFile
    
    try:
        with NamedTemporaryFile('wb', delete=False) as f:
            pickle.dump(data, f)
            tempname = f.name
        os.replace(tempname, path)
    except Exception as e:
        print(f"Failed to save checkpoint: {str(e)}")


def set_desc4tqdm(pbar, mode, n_items):
    pbar.set_description(
        f"Processing ({mode}, remaining {n_items} items)"
    )


def show_progress_bar(total:int, checkpoint_data:dict):

    widgets = [
        Percentage(), " ",
        Bar(), " ",
        Timer(format="%s"), " ",
        Counter(), f"/{total}"
    ]
    
    pbar = ProgressBar(widgets=widgets, maxval=total).start()
    while True:
        n_completed = len(checkpoint_data.get("progressed"))
        
        pbar.update(n_completed)

        if n_completed >= total:
            pbar.finish()
            break
        time.sleep(5)