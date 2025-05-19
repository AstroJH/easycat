import unittest
import time
import os
import pandas as pd
from easycat.parallel import dispatch

def sample_task(row: tuple) -> tuple:
    index, data = row
    time.sleep(2)
    if index == 5 and not os.environ.get("TEST_FORCE_STOP"):
        raise ValueError("模拟任务失败")
    return (index, {"data": data.to_dict(), "status": "success"})


class TestParallel(unittest.TestCase):
    @unittest.skip(reason="skip test_parallel")
    def test_parallel(self):
        df = pd.DataFrame({"value": range(5)})
        
        config = {
            "mode": "thread",
            "max_workers": 4,
            "checkpoint": "progress.pkl",
            "checkpoint_interval": 2
        }
        
        try:
            result = dispatch(
                df=df,
                process_func=sample_task,
                **config
            )
            print("\n处理完成:")
            print(result)
        except Exception as e:
            print(f"\n程序异常终止: {str(e)}")
        finally:
            if os.path.exists(config["checkpoint"]):
                os.remove(config["checkpoint"])

            