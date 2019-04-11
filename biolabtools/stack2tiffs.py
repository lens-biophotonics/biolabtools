import os
import math
import logging
import argparse

import numpy as np
import coloredlogs
import skimage.external.tifffile as tiff

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extracts individual frames from a 3D stack (all input '
                    'formats supported by Zetastitcher)',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_directory', type=str)

    parser.add_argument('-c', type=str, default=0, dest='compression',
                        choices=[str(i) for i in range(10)] + ['lzma'],
                        help='compression')

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

    os.makedirs(args.output_directory, exist_ok=True)

    n_of_digits = math.ceil(math.log10(infile.nfrms))
    output_filename_fmt = '{:0' + str(n_of_digits) + '}.tiff'

    for z in range(infile.nfrms):
        a = infile[z]
        if infile.nchannels != 1 and infile.channel == -1:
            a = np.moveaxis(a, -3, -1)
        output_filename = os.path.join(args.output_directory,
                                       output_filename_fmt.format(z))
        logger.info('saving to {}'.format(output_filename))
        tiff.imsave(output_filename, a, compress=args.compression)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
