import os.path
import logging
import argparse

import coloredlogs

import numpy as np
import skimage.external.tifffile as tiff


from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert directories of tiffs to tiff 3D.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n',
        formatter_class=CustomFormatter)

    parser.add_argument('input_path', type=str,
                        help='input path')
    parser.add_argument('output_dir', type=str, help='output directory')

    parser.add_argument('-c', type=int, default=-1, dest='channel',
                        help='channel')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    g = os.walk(args.input_path)

    flist = next(g)[1]

    for index in flist:
        index = os.path.join(args.input_path, index)
        logger.info(index)

        infile = InputFile(index)
        if args.channel != -1:
            infile.channel = args.channel

        a = infile.whole()

        if infile.nchannels != 1 and infile.channel == -1:
            a = np.moveaxis(a, -3, -1)

        name = os.path.basename(index)
        name, ext = os.path.splitext(name)
        oname = os.path.join(args.output_dir, name + '.tiff')

        tiff.imsave(oname, a)


if __name__ == '__main__':
    main()
