import os
import numpy as np
import pandas as pd
from easycat.lightcurve.reprocess import WISEReprocessor
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from config_jhwu import CONFIG

STORE_DIR = CONFIG.get("STORE_DIR")
REPRO_DIR = CONFIG.get("REPRO_DIR")

# 下载 WISE 光变数据所使用的源表
# 至少需要 obj_id, raj2000, dej2000 三列
catalog = pd.read_csv(CONFIG.get("CATALOG_PATH"))
catalog.set_index(keys="obj_id", inplace=True)

os.makedirs(REPRO_DIR, exist_ok=True)

repro = WISEReprocessor()
catalog_repro = pd.DataFrame(columns=catalog.columns) # 再处理后依然保留下来的源目录

# reprocessing data
MAX_INTERVAL = 10
for obj_id, row in catalog.iterrows():
    filepath = os.path.join(STORE_DIR, obj_id+".fits")
    if not os.path.isfile(filepath):
        print(f"WARNING: {obj_id} doesn't exist.")
        continue

    try:
        with fits.open(filepath) as hdul:
            lcurve = Table(hdul[1].data).to_pandas()

        pos_ref = SkyCoord(ra=row["raj2000"], dec=row["dej2000"], unit="deg", frame="fk5")

        lcurve_repro = repro.reprocess(
            lcurve=lcurve,
            pos_ref=pos_ref,
            dbscan=False,
            cone_radius=3 * u.arcsec,
            max_nb=2,
            max_rchi2=np.inf,
            outlier_threshold=5,
            max_interval=MAX_INTERVAL
        )

        lcurve_repro = repro.filter_uncertainty(lcurve_repro, w1threshold=np.inf, w2threshold=np.inf)

        lcurve_repro = repro.clean_epoch(lcurve_repro, n_least=5) # 每个 epoch 至少 5 个数据点
        longterm = repro.generate_longterm_lcurve(lcurve_repro, MAX_INTERVAL) # 构建长期光变曲线
    except Exception as e:
        print(f"ERROR: {obj_id} {e}")
        continue

    # at least 5 epochs
    if len(longterm) < 5: continue

    # save result
    catalog_repro.loc[obj_id] = row
    repropath = os.path.join(REPRO_DIR, obj_id+".csv")
    longterm.to_csv(repropath, index=False)

assert len(catalog_repro) <= len(catalog)
catalog_repro.to_csv(CONFIG.get("CATALOG_REPRO_PATH"), index=True, index_label="obj_id")