import logging
import argparse

import coloredlogs
import numpy as np
import skimage.external.tifffile as tiff


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply artificial HE coloring to two-photon images',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logger.info('Loading {}'.format(args.input_file))
    img = tiff.imread(args.input_file)
    img = np.moveaxis(img, 0, -1).astype(np.float32) / 255

    img_he = np.zeros_like(img)

    logger.info('applying HE coloring...')
    img_he[..., 0] = np.power(10, -(0.644 * img[..., 1] + 0.093 * img[..., 0]))
    img_he[..., 1] = np.power(10, -(0.717 * img[..., 1] + 0.954 * img[..., 0]))
    img_he[..., 2] = np.power(10, -(0.267 * img[..., 1] + 0.283 * img[..., 0]))

    logger.info('saving to {}'.format(args.output_file))
    tiff.imsave(args.output_file, (255 * img_he).astype(np.uint8))


if __name__ == '__main__':
    main()
