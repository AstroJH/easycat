import os
from os.path import join as pjoin
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from easycat.download import WISEDataArchive
from config_jhwu import CONFIG

STORE_DIR = CONFIG.get("STORE_DIR")
MATCH_RADIUS = CONFIG.get("MATCH_RADIUS")
N_WORKER = CONFIG.get("N_WORKER")
CHECKPOINT = CONFIG.get("CHECKPOINT")

catalog_name = "Paliya_2024_NLSy1"
catalog_fname = "seyfert-catalog-v1.fits"
filepath = pjoin("..", "catalog", catalog_fname)
with fits.open(filepath) as hdul:
    catalog = Table(hdul[1].data)
    catalog: pd.DataFrame = catalog.to_pandas()

catalog.rename(columns={
    "RA": "raj2000",
    "DEC": "dej2000",
    "Z": "z"
}, inplace=True)

filepath = pjoin("..", "catalog", catalog_fname)
catalog = pd.read_csv(filepath)
catalog.set_index("obj_id", inplace=True)

print("size:", len(catalog))

os.makedirs(STORE_DIR, exist_ok=True)

arch = WISEDataArchive()
record = arch.retrieve_photo(
    catalog     = catalog,
    return_data = False,
    radius      = MATCH_RADIUS,
    store_dir   = STORE_DIR,
    n_works     = N_WORKER,
    checkpoint  = CHECKPOINT
)

if len(record.get("completed")) == len(catalog):
    os.remove(CHECKPOINT)
