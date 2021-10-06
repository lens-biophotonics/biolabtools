import math
import logging
import os
import argparse

import coloredlogs
import tifffile as tiff

from zetastitcher import InputFile

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract tiffs from a stack',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', help='input file')
    parser.add_argument('output_dir', help='output directory')

    parser.add_argument('--zmin', type=int, default=0)
    parser.add_argument('--zmax', type=int)
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for output files')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger.info(args.input_file)
    infile = InputFile(args.input_file)

    os.makedirs(args.output_dir, exist_ok=True)

    n_of_digits = math.ceil(math.log10(infile.shape[0]))
    fmt = '{}{:0' + str(n_of_digits) + '}.tiff'

    zmax = infile.nfrms if args.zmax is None else args.zmax

    for z in range(args.zmin, zmax):
        fname = os.path.join(args.output_dir, fmt.format(args.prefix, z))
        logger.info(fname)
        tiff.imwrite(fname, infile[z], compression='zlib')


if __name__ == '__main__':
    main()
