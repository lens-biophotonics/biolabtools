import os.path
import logging
import argparse

import coloredlogs

from zetastitcher import InputFile
from zetastitcher import FileMatrix

import skimage.external.tifffile as tiff


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
    fm = FileMatrix(args.input_path)

    os.makedirs(args.output_dir, exist_ok=True)

    for index, row in fm.data_frame.iterrows():
        logger.info(index)

        infile = InputFile(index)
        if args.channel != -1:
            infile.channel = args.channel
        a = infile.whole()

        name = os.path.basename(index)
        name, ext = os.path.splitext(name)
        oname = os.path.join(args.output_dir, name + '.tiff')

        tiff.imsave(oname, a)


if __name__ == '__main__':
    main()
