import os
import math
import logging
import argparse

import coloredlogs

import numpy as np
import skimage.external.tifffile as tiff

from transforms3d import affines
from scipy import ndimage

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Reslice images taken with dual view SPIM',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('-t', '--theta', type=float, required=True,
                        help='rotation angle (degrees)')
    parser.add_argument('-r', '--ratio', type=float, required=True,
                        help='z / xy ratio')
    parser.add_argument('-v', '--view', type=str, required=True,
                        choices={'l', 'r'})
    parser.add_argument('-d', '--direction', type=str, required=True,
                        choices={'l', 'r'}, help='stage motion direction')
    parser.add_argument('-f', '--force', action='store_true',
                        help="don't stop if output file exists")
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    infile = InputFile(args.input_file)

    if os.path.exists(args.output_file):
        logger.warning('Output file {} already exists')
        if not args.force:
            return

    logger.info('loading {}'.format(args.input_file))
    a = infile.whole().T  # X, Y, Z order

    if args.direction == 'l':
        a = np.flip(a, -1)  # sample moving right to left

    r = args.ratio  # z / xy ratio
    theta = -args.theta * np.pi / 180
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    Dr = (a.shape[2] - 1) * r

    sheared_shape = np.array(a.shape, dtype=np.float64)
    sheared_shape[0] += Dr
    sheared_shape[2] = Dr

    final_shape = [
        a.shape[0] * abs(costheta) + sheared_shape[2] / abs(sintheta),
        a.shape[1],
        a.shape[0] * abs(sintheta),
        ]
    final_shape = np.ceil(np.abs(final_shape)).astype(np.int64)

    # transform to objective reference system
    T = [0, 0, 0]
    R = np.eye(3)
    Z = [1, 1, r]  # make voxel isotropic
    S = [0, r, 0]

    MO = affines.compose(T, R, Z, S)

    tempR = np.array([
        [costheta, 0, -sintheta, 0],
        [0, 1, 0, 0],
        [sintheta, 0, costheta, 0],
        [0, 0, 0, 1],
    ])

    center = sheared_shape / 2

    tempT = np.eye(4, 4)
    tempT[:3, -1] = center
    tempT2 = np.copy(tempT)
    tempT2[:3, -1] *= -1

    # rotation around Y axis relative to center
    R = np.linalg.multi_dot([tempT, tempR, tempT2])

    # final translation to center output in the "viewport"
    finalT = np.eye(4, 4)
    diff_shape = final_shape - sheared_shape
    finalT[:3, -1] = diff_shape / 2

    extraFlipZ = np.eye(4)
    extraFlipZ[2, -1] = final_shape[2]
    extraFlipZ[2, 2] = -1

    M = np.linalg.multi_dot([extraFlipZ, finalT, R, MO])
    M_inv = np.linalg.inv(M)

    # check that the 8 corners in the original stack are within the
    # transformed volume. Direction: forward (original -> transformed)
    yy, xx, zz = map(np.ndarray.flatten, np.meshgrid([0, 1], [0, 1], [0, 1]))
    coords = np.c_[xx, yy, zz] * (np.array(a.shape) - 1)
    coords = np.c_[coords, np.ones(8)]  # homogeneous coord

    transformed_coords = M.dot(coords.T).T.astype(np.int64)[:, :3]
    cond = (0 <= transformed_coords) & (transformed_coords < final_shape)
    if not cond.all():
        raise ValueError('Original corners outside transformed volume')

    # in the transformed space, iterate over the pixels around (0, 0, 0)
    # to find the extra translation needed to remove black voxels in that corner
    # Direction: backward (transformed -> original)
    rng = np.arange(0, 10)
    xx, yy, zz = map(np.ndarray.flatten, np.meshgrid(rng, rng, rng))
    coords = np.c_[xx, yy, zz]
    coords = np.c_[coords, np.ones(coords.shape[0])]  # homogeneous coord
    transformed_coords = M_inv.dot(coords.T).T[:, :3]

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (np.array(a.shape) - 1)))
    idx = np.all(cond, axis=-1)
    extraT = np.eye(4)
    extraT[:3, -1] = -1 * np.min(coords[idx], axis=0)[:3]

    M = np.linalg.multi_dot([extraT, extraFlipZ, finalT, R, MO])
    M_inv = np.linalg.inv(M)

    # in the transformed space, iterate over the pixels around the corner
    # opposite to the origin (0, 0, 0) to refine final_shape
    # Direction: backward (transformed -> original)

    # 3 is to preserve homogeneous coordinate
    coords = (np.r_[final_shape, 3] - 1 - coords)
    transformed_coords = M_inv.dot(coords.T).T[:, :3]

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (np.array(a.shape) - 1)))
    idx = np.all(cond, axis=-1)

    final_shape = (np.max(coords[idx, :3].astype(np.int64), axis=0) + 1)
    logger.info('input_shape: {}, output_shape: {}'
                .format(infile.shape, tuple(final_shape)))

    logger.info('applying transform...')
    transformed = ndimage.affine_transform(a, M_inv, output_shape=final_shape)

    if args.view == 'r':
        # cancel extra flip along Z
        transformed = np.flip(transformed, -1)

    # view sample from the front side
    transformed = np.flip(transformed, 0)

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    logger.info('saving to {}'.format(args.output_file))
    tiff.imsave(args.output_file, transformed.T)


if __name__ == '__main__':
    main()
