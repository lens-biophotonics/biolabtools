import os
import psutil
import logging
import argparse
from math import ceil

import coloredlogs
import numpy as np
import tifffile as tiff

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compute MIP (Maximum Intensity Projection) along the '
                    'outermost axis',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    group = parser.add_argument_group('input selection')
    parser.add_argument('-c', type=str, default=None, dest='channel',
                        choices=['r', 'g', 'b'], help='color channel')
    group.add_argument('--zmin', type=int, default=0, help='start frame')
    me_group = group.add_mutually_exclusive_group()
    me_group.add_argument('--zmax', type=int, help='end frame (noninclusive)')
    me_group.add_argument('--nz', type=int, help='number of z frames')

    args = parser.parse_args()

    channels = {
        'r': 0,
        'g': 1,
        'b': 2
    }

    if args.channel is not None:
        args.channel = channels[args.channel]

    if args.nz is not None:
        args.zmax = args.zmin + args.nz

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)

    if args.channel is not None:
        infile.channel = args.channel

    total_byte_size = np.asscalar(np.prod(infile.shape) * infile.dtype.itemsize)

    ram = psutil.virtual_memory().available * 0.5
    n_of_parts = ceil(total_byte_size / ram)

    try:
        os.remove(args.output_file)
    except FileNotFoundError:
        pass

    curr_z = 0
    if args.zmin is not None:
        curr_z = args.zmin
    nz = infile.nfrms
    if args.zmax is not None:
        nz = args.zmax - args.zmin
    zmax = curr_z + nz
    part_height = nz // n_of_parts
    end_loop = False

    mip = np.zeros(infile.shape[1:], dtype=infile.dtype)
    tiff.imwrite(args.output_file, mip)

    while True:
        end_z = curr_z + part_height
        if end_z >= zmax:
            end_z = zmax
            end_loop = True

        logger.info('loading \tz=[{}:{}]'.format(curr_z, end_z))
        img = infile[curr_z:end_z]

        logger.info('Computing MIP...')
        mip = np.maximum(np.max(img, axis=0), mip)

        del img

        curr_z = end_z
        if end_loop:
            break

    logger.info('saving to {}'.format(args.output_file))
    if args.channel is None and infile.nchannels > 1:
        mip = np.moveaxis(mip, -3, -1)
    tiff.imwrite(args.output_file, mip, compression='zlib')


if __name__ == '__main__':
    main()
