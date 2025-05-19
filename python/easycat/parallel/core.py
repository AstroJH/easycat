import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import signal
import os
from typing import Callable, Optional
import time

class GracefulInterrupt:
    """Graceful interrupt handler"""

    def __init__(self):
        self.interrupted = False
        self.original_sigint = signal.getsignal(signal.SIGINT)
        
    def __enter__(self):
        """Register signal handler"""
        signal.signal(signal.SIGINT, self._handler)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original signal handler"""
        signal.signal(signal.SIGINT, self.original_sigint)
        
    def _handler(self, signum, frame):
        """Custom signal handler"""
        if self.interrupted:
            print("\nForceful termination prevented. Please wait for current tasks to complete...")
            return
            
        self.interrupted = True
        print("\nInterrupt request detected. Stopping processing... (Press Ctrl+\\ to force quit)")

def dispatch(
    df: pd.DataFrame,
    process_func: Callable,
    mode: str = "thread",
    max_workers: Optional[int] = None,
    checkpoint: Optional[str] = None,
    checkpoint_interval: int = 10
) -> pd.DataFrame:
    """
    Parallel processing with graceful interruption and resume support
    
    Args:
        df: Input DataFrame to process
        process_func: Processing function that accepts (index, row) tuple
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
    checkpoint_data = {"processed": set(), "results": {}}
    
    # Load existing checkpoint
    if checkpoint and os.path.exists(checkpoint):
        import pickle
        with open(checkpoint, "rb") as f:
            checkpoint_data = pickle.load(f)
        print(f"Checkpoint file detected. Resumed {len(checkpoint_data['processed'])} records")

    # Create executor
    Executor = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor
    max_workers = max_workers or (os.cpu_count() or 4)
    
    with GracefulInterrupt() as interrupt, Executor(max_workers=max_workers) as executor:
        # Submit tasks (skip completed)
        futures = {}
        for index, row in df.iterrows():
            if index not in checkpoint_data["processed"]:
                future = executor.submit(process_func, (index, row))
                futures[future] = index
                
        # Progress bar configuration
        total = len(futures)
        last_save = time.time()
        desc = f"Processing ({mode}, remaining {total} items)"
        
        with tqdm(total=total, desc=desc) as pbar:
            try:
                for future in as_completed(futures):
                    if interrupt.interrupted:
                        break
                        
                    # Process result
                    index = futures[future]
                    try:
                        result = future.result()
                        results[index] = result
                        checkpoint_data["processed"].add(index)
                        checkpoint_data["results"][index] = result
                    except Exception as e:
                        print(f"\nTask {index} failed: {str(e)[:200]}")
                        
                    # Update progress
                    pbar.update(1)
                    pbar.set_description(f"Processing ({mode}, remaining {len(futures)-pbar.n} items)")
                    
                    # Periodic checkpoint save
                    if checkpoint and (time.time() - last_save) > checkpoint_interval:
                        _save_checkpoint(checkpoint, checkpoint_data)
                        last_save = time.time()
                        
            except KeyboardInterrupt:
                if not interrupt.interrupted:
                    raise  # Handle non-signal interrupts
                    
        # Final checkpoint save
        if checkpoint:
            _save_checkpoint(checkpoint, checkpoint_data)
            
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