import os
import logging
import argparse

import coloredlogs
import numpy as np
import tifffile as tiff

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply artificial HE coloring to two-photon images',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    parser.add_argument('-n', type=int, default=1,
                        help='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)
    infile.squeeze = False

    total_byte_size = np.asscalar(np.prod(infile.shape) * infile.dtype.itemsize)
    bigtiff = total_byte_size > 2 ** 31 - 1

    try:
        os.remove(args.output_file)
    except FileNotFoundError:
        pass

    curr_z = 0
    part_height = infile.nfrms // args.n
    end_loop = False

    while True:
        end_z = curr_z + part_height
        if end_z >= infile.shape[0]:
            end_z = infile.shape[0]
            end_loop = True

        logger.info('loading \tz=[{}:{}]'.format(curr_z, end_z))
        img = infile[curr_z:end_z]
        img = img.astype(np.float32) / 255
        img_he = np.zeros_like(img)

        logger.info('applying HE coloring...')
        idx_0 = np.index_exp[:, 0, ...]
        idx_1 = np.index_exp[:, 1, ...]
        idx_2 = np.index_exp[:, 2, ...]
        img_he[idx_0] = np.power(10, -(0.644 * img[idx_1] + 0.093 * img[idx_0]))
        img_he[idx_1] = np.power(10, -(0.717 * img[idx_1] + 0.954 * img[idx_0]))
        img_he[idx_2] = np.power(10, -(0.267 * img[idx_1] + 0.283 * img[idx_0]))

        logger.info('saving to {}'.format(args.output_file))
        if infile.nchannels > 1:
            img_he = np.moveaxis(img_he, -3, -1)
        tiff.imwrite(args.output_file, (255 * img_he).astype(np.uint8),
                     append=True, bigtiff=bigtiff)

        curr_z = end_z
        if end_loop:
            break


if __name__ == '__main__':
    main()
