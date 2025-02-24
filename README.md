# Easy Cat

## Common Tool
```Python
import easycat

def handler(obj_id:str, params:dict) -> bool:
    pass

easycat.start(logpath, catalog, handler, n_workers=10)
```

## Download WISE Data
### Case1: Single infrared object
```Python
from easycat.download import WISEDataDownloader
from astropy import units as u

downloader = WISEDataDownloader(radius=3*u.arcsecond, store_dir="./wise_data/")
is_successful = downloader.download("filename", raj2000=114.2125, dej2000=65.6025)
```

### Case2: Multiple infrared objects
```Python
import pandas as pd
from easycat.download import WISEDataDownloader
from astropy import units as u

catalog = pd.DataFrame(
    data={
        "obj_id": ["A000", "B001", "C101"],
        "raj2000": [0.047549, 0.09689882, 0.2813572],
        "dej2000": [14.92935, 19.45895, 0.4404623],
        # More columns ...
    },
)
catalog.set_index(keys="obj_id", inplace=True)

downloader = WISEDataDownloader(radius=3*u.arcsecond, store_dir="./test_store_dir",
                                catalog=catalog, logpath="./test.log.npy")
downloader.download()
```

> **Tip**
> You can create the `catalog` by reading from Excel Table or other local file.