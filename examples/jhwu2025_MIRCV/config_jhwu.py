from os.path import join as pjoin
from astropy import units as u

catalog_path       = pjoin("data", "Paliya_2024_NLSy1.csv")
catalog_repro_path = pjoin("data", "Paliya_2024_NLSy1_repro.csv")
store_dir          = pjoin("data", "Paliya_2024_NLSy1")
repro_dir          = pjoin("data", "Paliya_2024_NLSy1_repro")
checkpoint         = pjoin("data", "wisedata_checkpoint_Paliya_2024_NLSy1.pkl")

CONFIG = {
    "CATALOG_PATH": catalog_path,
    "CATALOG_REPRO_PATH": catalog_repro_path,
    "STORE_DIR": store_dir, # WISE 光变数据所存放的目录
    "REPRO_DIR": repro_dir, # 再处理后的数据需要存放的目录
    "CHECKPOINT": checkpoint,

    "ERR_SYS_W1": 0.024,
    "ERR_SYS_W2": 0.028,
}