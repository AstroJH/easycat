import argparse
from ..astro_xray.simple_grppha import grppha

def simple_grppha() -> None:
    parser = argparse.ArgumentParser(description="Simple grppha.")

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-m", "--mincts", required=True, type=float)
    parser.add_argument("-b", "--bkg", default=None)
    parser.add_argument("-r", "--rmf", default=None)
    parser.add_argument("-n", "--netreg", default="no", choices=["yes", "no"])
    parser.add_argument("--notice", type=float, nargs='+', default=())
    parser.add_argument("--clobber", default="no", choices=["yes", "no"])

    args = parser.parse_args()

    netreg = args.netreg == "yes"
    clobber = args.clobber == "yes"

    grppha(args.input, args.output, args.mincts, args.notice,
            netreg=netreg, fn_rmf=args.rmf, fn_spec_bkg=args.bkg, overwrite=clobber)