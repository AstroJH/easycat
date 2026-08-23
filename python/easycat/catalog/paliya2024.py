from astropy.table import Table, vstack
from astropy.io import fits
from pathlib import Path

def load(cfg, **kwargs):
    load_cfg = cfg["load"]
    filepath = Path(load_cfg["target"])
    with fits.open(filepath) as hdul:
        return Table(hdul[1].data)

def build(cfg, *, overwrite: bool=False, **kwargs):
    build_cfg = cfg["build"]

    filepath = Path(build_cfg["raw"])

    with fits.open(filepath) as hdul:
        nls = Table(hdul["NLSy1"].data)
        bls = Table(hdul["BLSy1"].data)

    # Add type
    nls["type"] = "NLSy1"
    bls["type"] = "BLSy1"

    # Merge
    table: Table = vstack([nls, bls])

    # Generate OBJ_ID
    obj_id_lst = [
        f"paliya2024_{i}"
        for i in range(len(table))
    ]

    table.add_column(col=obj_id_lst, name=build_cfg["id"], index=0)

    # Save table data
    output = Path(build_cfg["target"])
    if output.exists() and not overwrite:
        print(f"`{output}` exists. Skip saving.")
        return False

    print(f"Saving to `{output}`")
    write_catalog(table, output, overwrite=overwrite)
    print("Ok.\n")

    # Print log
    print(
        f"Built catalog: {len(table)} objects, "
        f"{len(table.colnames)} fields"
    )

    print("First 5 rows:")
    table[:5].pprint()
    print()

    return True


def read_catalog(filename):
    filename = Path(filename)
    return Table.read(filename)


def write_catalog(catalog, output, *, overwrite=False):
    output = Path(output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    catalog.write(output, overwrite=overwrite)