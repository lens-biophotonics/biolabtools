import sys
import psutil
import logging
import argparse

from pathlib import Path

import numpy as np
import coloredlogs
import tifffile as tiff

from zetastitcher import InputFile
from zetastitcher.io.tiffwrapper import TiffWrapper


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converts an input file to a 3D TIFF (all input formats supported by Zetastitcher)',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)

    parser.add_argument('-c', type=str, default=0, dest='compression',
                        choices=[str(i) for i in range(10)] + ['lzma'],
                        help='compression')
    parser.add_argument('-f', action='store_true', help='force overwriting output file')

    parser.add_argument('--ch', type=int, default=-1, dest='channel',
                        help='channel')

    args = parser.parse_args()

    try:
        args.compression = int(args.compression)
    except ValueError:
        pass

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)
    if args.channel != -1:
        infile.channel = args.channel

    if type(infile.wrapper) is TiffWrapper:
        if infile.wrapper.glob_mode is False:
            raise ValueError('Input file is already a TIFF')

    output_file = Path(args.input_file).with_suffix('.tiff')
    if output_file.exists() and not args.f:
        logger.error(f'Output file {output_file} already exists. Use -f to force.')
        sys.exit(-1)

    bigtiff = infile.array_size > 2 ** 31 - 1

    ram = psutil.virtual_memory().available

    # size in bytes of an xy plane (including channels) (float32)
    n_frames_in_ram = int(ram / infile.frame_size / 1.8)

    n_loops = infile.nfrms // n_frames_in_ram

    partial_thickness = [n_frames_in_ram for _ in range(0, n_loops)]
    remainder = infile.nfrms % n_frames_in_ram
    if remainder:
        partial_thickness += [remainder]

    try:
        output_file.unlink()
    except FileNotFoundError:
        pass

    z = 0
    for thickness in partial_thickness:
        logger.info(f'loading {args.input_file} z=[{z}:{z + thickness}]')
        a = infile.zslice(z, z + thickness)
        z += thickness

        if infile.nchannels > 1:
            a = np.moveaxis(a, -3, -1)

        logger.info(f'saving output to {output_file}')
        tiff.imwrite(output_file, a, append=True, bigtiff=bigtiff, compress=args.compression)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
