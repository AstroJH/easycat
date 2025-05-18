import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import requests
import os
from typing import Callable

def parallel_process_dataframe(
    df: pd.DataFrame,
    process_func: Callable,
    mode: str = "thread",
    max_workers: int = None,
    chunksize: int = None,
) -> pd.DataFrame:
    """
    并行处理DataFrame的核心函数

    Parameters
    ==========
        df: 要处理的DataFrame
        process_func: 处理函数，接收(index, row)元组
        mode: 并行模式 ("thread"|"process")
        max_workers: 最大工作线程/进程数
        chunksize: 进程池的任务分块大小（仅进程模式有效）

    Return
    ======
        包含处理结果的DataFrame, 保持原始顺序
    """
    # 参数校验
    if mode not in ("thread", "process"):
        raise ValueError("Invalid mode. Choose 'thread' or 'process'")

    # 设置默认max_workers
    if max_workers is None:
        max_workers = os.cpu_count() or 4
        if mode == "thread":
            max_workers = min(max_workers * 4, 32)  # I/O密集型任务使用更多线程

    # 选择执行器类型
    Executor = ThreadPoolExecutor if mode == "thread" else ProcessPoolExecutor

    results = []
    with Executor(max_workers=max_workers) as executor:
        # 提交任务
        futures = {
            executor.submit(process_func, (index, row)): index
            for index, row in df.iterrows()
        }

        # 初始化进度条
        with tqdm(total=len(df), desc=f"Processing ({mode})") as pbar:
            # 按完成顺序收集结果
            unordered_results = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    unordered_results.append(result)
                except Exception as e:
                    print(f"\nTask failed: {str(e)[:200]}")
                finally:
                    pbar.update(1)

    # 按原始索引排序结果
    ordered_results = sorted(unordered_results, key=lambda x: x[0])
    
    # 构建结果DataFrame
    result_df = pd.DataFrame([r[1] for r in ordered_results])
    result_df.index = [r[0] for r in ordered_results]
    return result_df.sort_index()


def download_task(row: tuple) -> tuple:
    ...

if __name__ == "__main__":
    # 示例数据
    data = {
        "url": [f"https://httpbin.org/delay/{i}" for i in [1, 2, 3, 1, 2]],
        "param": list("abcde")
    }
    df = pd.DataFrame(data)
    
    # 运行参数选择
    mode = input("Choose mode [thread/process]: ").strip().lower()
    
    # 执行处理
    result_df = parallel_process_dataframe(
        df=df,
        process_func=download_task,
        mode=mode,
        max_workers=5
    )
    
    # 显示结果
    print("\nProcessed results:")
    print(result_df[["url", "param", "status"]].head())