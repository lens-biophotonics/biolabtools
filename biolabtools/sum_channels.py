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
        description='Sum RGB channels to 16 bit grayscale image',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)

    if infile.nchannels <= 1:
        raise ValueError('Not enough channels: {}'.format(infile.nchannels))

    # input
    total_byte_size = np.asscalar(np.prod(infile.shape) * infile.dtype.itemsize)
    n = int(total_byte_size // (psutil.virtual_memory().available * 0.25)) + 1

    # output
    total_byte_size = infile.nfrms * infile.ysize * infile.xsize * 2
    bigtiff = total_byte_size > 2 ** 31 - 1

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

        img = np.sum(img, axis=1).astype(np.uint16)

        logger.info('saving to {}'.format(args.output_file))
        tiff.imwrite(args.output_file, img, append=True, bigtiff=bigtiff, compression='zlib')

        curr_z = end_z
        if end_loop:
            break


if __name__ == '__main__':
    main()
