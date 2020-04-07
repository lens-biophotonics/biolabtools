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
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n'
               '        Vladislav Gavryusev <gavryusev@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter,
    )

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
    parser.add_argument('-s', '--slices', metavar='N', type=int, required=False,
                        help='perform transform in N slices, one at a time '
                             'to reduce memory requirements')
    parser.add_argument('input_file', type=str)
    parser.add_argument('output_file', type=str)

    args = parser.parse_args()

    return args


def grid_to_coords(*xi):
    xx, yy, zz = map(np.ndarray.flatten, np.meshgrid(*xi))
    coords = np.c_[xx, yy, zz]
    return coords


def transform_coords(M, coords):
    coords = np.array(coords)
    if coords.ndim == 1:
        coords = coords[np.newaxis, ...]
    if coords.shape[1] == 3:
        coords = np.c_[coords, np.ones(coords.shape[0])]  # homogeneous coord

    return M.dot(coords.T).T[:, :3]


def inv_matrix(shape, theta, r):
    """
    Compute inverse coordinate transformation matrix.

    Parameters
    ----------
    shape : tuple
            input shape, in (X, Y, Z) order
    theta : float
            rotation angle (degrees)
    r : float
        z / xy ratio

    Returns
    -------
    M_inv : :class:`numpy.ndarray`
            The inverse coordinate transformation inv_matrix, mapping output
            coordinates to input coordinates. To be used in
            :func:`scipy.ndimage.affine_transform`
    output_shape : tuple
            The shape of the transformed output, in (X, Y, Z) order
    """
    shape = np.array(shape)

    theta = -theta * np.pi / 180
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    Dr = (shape[2] - 1) * r

    sheared_shape = np.array(shape, dtype=np.float64)
    sheared_shape[0] += Dr
    sheared_shape[2] = Dr

    final_shape = [
        shape[0] * abs(costheta) + sheared_shape[2] / abs(sintheta),
        shape[1],
        shape[0] * abs(sintheta),
    ]
    final_shape = np.ceil(np.abs(final_shape)).astype(np.int64)

    # transform to objective reference system
    T = [0, 0, 0]
    R = np.eye(3)
    Z = [1, 1, r]  # make voxel isotropic
    S = [0, r * abs(math.tan(theta)), 0]

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
    xx, yy, zz = map(np.ndarray.flatten, np.meshgrid([0, 1], [0, 1], [0, 1]))
    coords = grid_to_coords([0, 1], [0, 1], [0, 1]) * (shape - 1)
    transformed_coords = transform_coords(M, coords).astype(np.int64)

    cond = (0 <= transformed_coords) & (transformed_coords < final_shape)
    if not cond.all():
        raise ValueError('Original corners outside transformed volume')

    # in the transformed space, iterate over the pixels around (0, 0, 0)
    # to find the extra translation needed to remove black voxels in that corner
    # Direction: backward (transformed -> original)
    rng = np.arange(0, 10)
    coords = grid_to_coords(rng, rng, rng)
    transformed_coords = transform_coords(M_inv, coords)

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (shape - 1)))
    idx = np.all(cond, axis=-1)
    extraT = np.eye(4)
    extraT[:3, -1] = -1 * np.min(coords[idx], axis=0)[:3]

    M = np.linalg.multi_dot([extraT, extraFlipZ, finalT, R, MO])
    M_inv = np.linalg.inv(M)

    # in the transformed space, iterate over the pixels around the corner
    # opposite to the origin (0, 0, 0) to refine final_shape
    # Direction: backward (transformed -> original)

    # 3 is to preserve homogeneous coordinate
    coords = np.c_[coords, np.ones(coords.shape[0])]  # homogeneous coord
    coords = (np.r_[final_shape, 3] - 1 - coords)
    transformed_coords = transform_coords(M_inv, coords)

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (shape - 1)))
    idx = np.all(cond, axis=-1)

    final_shape = tuple(np.max(coords[idx, :3].astype(np.int64), axis=0) + 1)

    return M_inv, final_shape


def transform(array, M_inv, output_shape, view, offset=None):
    temp_M_inv = np.copy(M_inv)
    if offset is not None:
        temp_M_inv[:3, -1] += offset

    logger.info('applying transform...')
    transformed = ndimage.affine_transform(
        array, temp_M_inv, output_shape=output_shape)

    if view == 'r':
        # cancel extra flip along Z
        transformed = np.flip(transformed, -1)

    # view sample from the front side
    transformed = np.flip(transformed, 0)

    return transformed


def sliced_transform(array, M_inv, final_shape, view, n=8):
    final_shape = np.array(final_shape)

    stripe_height = final_shape[-1] // n
    remainder = final_shape[-1] % n

    top_edges = np.linspace(0, (n - 1) * stripe_height, n)
    heights = [stripe_height] * len(top_edges)
    heights[-1] += remainder

    current_z = 0
    offset_offset = np.squeeze(transform_coords(M_inv, [0, 0, 0]))
    for h in heights:
        output_shape = np.copy(final_shape)
        output_shape[-1] = h

        offset = transform_coords(M_inv, [0, 0, current_z])
        offset = np.squeeze(offset) - offset_offset

        yield transform(array, M_inv, output_shape, view, offset)
        current_z += h


def main():
    args = parse_args()

    infile = InputFile(args.input_file)
    ashape = np.flipud(np.array(infile.shape))  # X, Y, Z order

    M_inv, final_shape = inv_matrix(ashape, args.theta, args.ratio)

    logger.info('input_shape: {}, output_shape: {}'
                .format(infile.shape, tuple(final_shape)))

    if os.path.exists(args.output_file):
        logger.error('Output file {} already exists (use -f to force)')
        if not args.force:
            return

    logger.info('loading {}'.format(args.input_file))
    a = infile.whole().T  # X, Y, Z order

    if args.direction == 'l':
        a = np.flip(a, -1)  # sample moving right to left

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_byte_size = np.asscalar(np.prod(final_shape) * infile.dtype.itemsize)
    bigtiff = total_byte_size > 2 ** 31 - 1

    if args.slices is None:
        t = transform(a, M_inv, final_shape, args.view)
        logger.info('saving to {}'.format(args.output_file))
        tiff.imsave(args.output_file, t.T, bigtiff=bigtiff)
        return

    if os.path.exists(args.output_file):
        os.remove(args.output_file)

    i = 0
    for t in sliced_transform(a, M_inv, final_shape, args.view, args.slices):
        i += 1
        logger.info('saving slice {}/{} to {}'.format(
            i, args.slices, args.output_file))

        t = t.T  # Z, Y, X order

        # add dummy color axis to trick imsave
        # (otherwise when size of Z is 3, it thinks it's an RGB image)
        t = t[:, np.newaxis, ...]
        tiff.imsave(args.output_file, t, append=True, bigtiff=bigtiff)


if __name__ == '__main__':
    main()
