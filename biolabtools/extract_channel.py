import os
import logging
import argparse

import coloredlogs
import numpy as np
import tifffile as tiff

import psutil

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


channel_mapping = {
    'r': 0,
    'g': 1,
    'b': 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract a single channel',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    parser.add_argument('-c', type=str, choices=('r', 'g', 'b'), required=True, help='channel')

    args = parser.parse_args()
    setattr(args, 'c', channel_mapping[args.c])

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)
    infile.channel = args.c

    total_byte_size = np.asscalar(np.prod(infile.shape) * infile.dtype.itemsize)
    bigtiff = total_byte_size > 2 ** 31 - 1

    n = int(total_byte_size // (psutil.virtual_memory().available * 0.9)) + 1

    try:
        os.remove(args.output_file)
    except FileNotFoundError:
        pass

    curr_z = 0
    part_height = infile.nfrms // n
    end_loop = False

    while True:
        end_z = curr_z + part_height
        if end_z >= infile.shape[0]:
            end_z = infile.shape[0]
            end_loop = True

        logger.info('loading \tz=[{}:{}]'.format(curr_z, end_z))
        img = infile[curr_z:end_z]

        logger.info('saving to {}'.format(args.output_file))
        tiff.imsave(args.output_file, img, append=True, bigtiff=bigtiff)

        curr_z = end_z
        if end_loop:
            break


if __name__ == '__main__':
    main()
