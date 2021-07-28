import logging
import argparse

import coloredlogs
import tifffile as tiff

from zetastitcher import InputFile


logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Flip stack along Z',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    return args


def main():
    coloredlogs.install(level='INFO',
                        fmt='%(levelname)s [%(name)s]: %(message)s')

    args = parse_args()

    logger.info('loading {}'.format( args.input_file))
    infile = InputFile(args.input_file)

    a = infile.whole()
    a = a[::-1]

    logger.info('writing to {}'.format(args.output_file))
    tiff.imsave(args.output_file, a)


if __name__ == '__main__':
    main()
