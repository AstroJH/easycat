import pandas as pd
import os

from astropy import units as u

from easycat.download import WISEDataArchive


store_dir = "./wisedata/"
os.makedirs(store_dir, exist_ok=True)

arch = WISEDataArchive()

catalog = pd.read_csv("test.csv")
catalog.rename(columns={
    "RA": "raj2000",
    "Dec": "dej2000"
}, inplace=True)
catalog.set_index(keys="Name", inplace=True)


record = arch.retrieve_photo(
    catalog=catalog,
    radius=6*u.arcsec,

    store_dir=store_dir,
    return_data=False,
    n_works=3,
    # checkpoint="wisedata_checkpoint.pkl"
)