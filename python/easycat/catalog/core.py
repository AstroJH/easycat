from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import importlib
from agnkit.config import load_config

class Catalog:

    def __init__(self, table, cfg):
        # if isinstance(table, Table):
        #     table = table.to_pandas()

        self.table = table
        self.cfg = cfg


    @staticmethod
    def load(catname):
        cfg = load_config(f'{catname}.yaml', cfg_type="catalog")

        module = importlib.import_module(
            cfg["module"]
        )
        table = module.load(cfg)

        cat = Catalog(table, cfg)
        return cat

    def by_id(self, obj_id):
        mask = (
            self.table["obj_id"]
            ==
            obj_id
        )

        result = self.table[mask]

        if len(result):
            return result[0]

        return None

    def by_skycoord(self, target: SkyCoord, radius):
        table = self.table
        coords = SkyCoord(
            table["RA"],
            table["DEC"],
            unit="deg"
        )

        idx, sep, _ = target.match_to_catalog_sky(coords)

        if sep < radius:
            return table[idx:idx+1]
    
        return None

    def by_coord(self, ra, dec, radius=3*u.arcsec):
        target = SkyCoord(ra, dec, unit="deg")
        return self.by_skycoord(target, radius)

    def by_name(self, name, radius=3*u.arcsec, parse: bool=False):
        target = SkyCoord.from_name(name, parse=parse)
        return self.by_skycoord(target, radius)


def build_catalog(config, **kwargs):
    module = importlib.import_module(
        config["module"]
    )
    return module.build(config, **kwargs)
