"""Easycat astronomical data analysis toolkit."""

__all__ = [
    "download",
    "subcat2d",
    "lightcurve",
    "parallel",
    "stats",
    "astrofilter"
]

def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")