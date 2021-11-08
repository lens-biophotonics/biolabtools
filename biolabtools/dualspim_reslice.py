import os
import math
import time
import logging
import argparse
import threading
from queue import Queue
from pathlib import Path

import coloredlogs

import numpy as np
import tifffile as tiff

from scipy import ndimage

from zetastitcher import InputFile

from biolabtools.convert_to_jp2ar import convert_to_jp2ar


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
    parser.add_argument('-z', type=float, default=1,
                        help='zoom factor along z')
    parser.add_argument('-xy', type=float, default=1,
                        help='zoom factor along xy')
    parser.add_argument('-v', '--view', type=str, required=True,
                        choices={'l', 'r'})
    parser.add_argument('-d', '--direction', type=str, required=True,
                        choices={'l', 'r'}, help='stage motion direction')
    parser.add_argument('-f', '--force', action='store_true',
                        help="don't stop if output file exists")
    parser.add_argument('-s', '--slices', metavar='N', type=int, required=False,
                        help='perform transform in N slices, one at a time '
                             'to reduce memory requirements')
    parser.add_argument('-o', '--output_dir', type=str)

    group = parser.add_argument_group('JPEG2000 Archive options')
    group.add_argument('--ja', action='store_true', dest='jp2ar_enabled',
                       help='enable conversion to JPEG2000 ZIP archives')
    group.add_argument('-n', type=int, default=8, dest='nthreads',
                       help='number of parallel threads')
    group.add_argument('--jc', type=int, default=0, dest='jp2_compression',
                       help='JPEG2000 compression')

    parser.add_argument('input_file', type=str)

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


def inv_matrix(shape, theta, direction, view, z=1, xy=1):
    """
    Compute inverse coordinate transformation matrix.

    Parameters
    ----------
    shape : sequence
            input shape, in (X, Y, Z) order
    theta : float
            rotation angle (degrees)
    z : float
        zoom factor along z

    xy : float
        zoom factor along xy

    direction : str
                stage motion direction ('l' or 'r')

    view : str
           camera view ('l' or 'r')

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
    edge = shape - 1

    theta = -theta * np.pi / 180
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    # transform to objective reference system
    # zoom matrix
    Z = [xy, xy, z, 1] * np.eye(4)

    S = np.eye(4)  # shear matrix
    S[0, 2] = 1 / abs(math.tan(theta))

    M_list = [S, Z]
    M = np.linalg.multi_dot(M_list)

    corners = grid_to_coords([0, edge[0]], [0, edge[1]], [0, edge[2]])
    tr_corners = transform_coords(M, corners)
    sheared_size = np.max(tr_corners, axis=0) - np.min(tr_corners, axis=0)

    tempR = np.array([
        [costheta, 0, -sintheta, 0],
        [0, 1, 0, 0],
        [sintheta, 0, costheta, 0],
        [0, 0, 0, 1],
    ])

    center = sheared_size / 2

    tempT = np.eye(4, 4)
    tempT[:3, -1] = center
    tempT2 = np.copy(tempT)
    tempT2[:3, -1] *= -1

    # rotation around Y axis relative to center
    R = np.linalg.multi_dot([tempT, tempR, tempT2])

    M_list = [R] + M_list

    M = np.linalg.multi_dot(M_list)

    tr_corners = transform_coords(M, corners[[0, 3]])
    final_size = np.max(tr_corners, axis=0) - np.min(tr_corners, axis=0)
    final_shape = final_size.astype(np.int64)
    final_shape[1] = shape[1]

    # final translation to center output in the "viewport"
    finalT = np.eye(4, 4)
    finalT[:3, -1] = -1 * np.min(tr_corners, axis=0)

    M_list = [finalT] + M_list

    M = np.linalg.multi_dot(M_list)
    M_inv = np.linalg.inv(M)

    tr_corners = transform_coords(M, corners[[0, -1]])
    tr_corners -= np.min(tr_corners, axis=0)
    temp_corners = np.ceil(tr_corners)

    final_size = np.max(temp_corners, axis=0)
    final_shape = final_size.astype(np.int64)
    if xy == 1:
        final_shape[1] = shape[1]

    # correct crop at corners
    # =======================
    rng = np.arange(0, 10)

    coords = grid_to_coords(rng, 0, final_shape[2] - 1 - rng)
    transformed_coords = transform_coords(M_inv, coords)

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (shape - 1)))
    idx = np.all(cond, axis=-1)

    if idx.any():
        m = transformed_coords[idx].min(axis=0)
        valid = (0 <= m) & (m <= shape - 1)
        correction = m
        if valid.any():
            correction[valid] = 0

        M[:3, -1] += correction
        M_inv = np.linalg.inv(M)

    transformed_coords = transform_coords(M_inv, coords)

    cond = ((0 <= transformed_coords)
            & (transformed_coords <= (shape - 1)))
    idx = np.all(cond, axis=-1)

    shift_X = 0
    shift_Z = 0
    if idx.any():
        shift_X = np.min(coords[idx, 0])
        shift_Z = final_shape[2] - 1 - np.max(coords[idx, 2])

    M[0, -1] += -shift_X
    M[2, -1] += shift_Z
    final_shape[0] += -shift_X
    final_shape[2] += -shift_Z

    # =======================

    # Combine M with all other flips

    initial_flip_Z = np.eye(4)
    if direction == 'l':  # sample moving from right to left
        initial_flip_Z[2, -1] = edge[2]
        initial_flip_Z[2, 2] = -1

    final_flip_Z = np.eye(4)
    if view == 'l':
        final_flip_Z[2, -1] = final_shape[2] - 1
        final_flip_Z[2, 2] = -1

    # view sample from the front side
    flip_X = np.eye(4)
    flip_X[0, -1] = final_shape[0] - 1
    flip_X[0, 0] = -1

    M = np.linalg.multi_dot([flip_X, final_flip_Z, M, initial_flip_Z])
    M_inv = np.linalg.inv(M)

    return M_inv, final_shape


def transform(array, M_inv, output_shape, offset=None):
    """
    Apply affine transform to input array.

    Parameters
    ----------
    array : :class:`numpy.ndarray`
            Input array.
    M_inv : :class:`numpy.ndarray`
            The inverse coordinate transformation matrix (4x4,
            using homogeneous coordinates).
    output_shape : tuple
    offset : sequence
             Given an output image pixel index vector `o`, the pixel value is
             determined from the input image at position
             `np.dot(matrix, o) + offset`.

    Returns
    -------
    transformed : :class:`numpy.ndarray`
                  The transformed input.
    """
    temp_M_inv = np.copy(M_inv)
    if offset is not None:
        temp_M_inv[:3, -1] += offset

    logger.info('applying transform...')
    t = time.time()
    transformed = ndimage.affine_transform(
        array, temp_M_inv, output_shape=output_shape, order=0,
        prefilter=False)
    logger.info('transform done. Took: {:.1f}s'.format(time.time() - t))

    return transformed


def sliced_transform(array, M_inv, output_shape, n=8):
    """
    A generator calling :func:`transform` for each slice. This is useful to
    compute smaller transforms and use less RAM.

    The transformation is applied in slices along Z in the transformed space,
    which map to slices along X in the input space.

    Parameters
    ----------
    array : :class:`numpy.ndarray`
            Input array.
    M_inv : :class:`numpy.ndarray`
            The inverse coordinate transformation matrix (4x4,
            using homogeneous coordinates).
    output_shape : tuple
    n : int
        Number of slices.

    Yields
    -------
    transformed : :class:`numpy.ndarray`
              The transformed slice.
    """
    q = Queue(maxsize=1)
    t = threading.Thread(target=_sliced_transform, args=[
        q, array, M_inv, output_shape, n
    ])
    t.start()

    while True:
        got = q.get()
        if got is None:
            break
        yield got

    t.join()


def _sliced_transform(q, array, M_inv, output_shape, n=8):
    output_shape = np.array(output_shape)

    stripe_height = output_shape[-1] // n

    top_edges = np.linspace(0, n * stripe_height, n + 1)
    bottom_edges = top_edges + stripe_height - 1
    bottom_edges[-1] = output_shape[2] - 1
    coords_from = grid_to_coords(0, 0, top_edges)
    coords_to = grid_to_coords(output_shape[0] - 1, output_shape[1] - 1,
                               bottom_edges)
    offsets_from = transform_coords(M_inv, coords_from)
    offsets_to = transform_coords(M_inv, coords_to)

    heights = bottom_edges - top_edges + 1

    current_z = 0
    offset_offset = np.squeeze(transform_coords(M_inv, [0, 0, 0]))
    for h, o_from, o_to in zip(heights, offsets_from, offsets_to):
        temp_shape = np.copy(output_shape)
        temp_shape[-1] = h
        temp_shape = tuple(temp_shape)

        coords = np.c_[o_from, o_to]

        coords_min = np.min(coords, axis=1)
        coords_max = np.max(coords, axis=1)

        coords_min[coords_min < 0] = 0
        idx = coords_max < 0
        coords_max[idx] = output_shape[idx]
        coords_min = coords_min.astype(np.int64)

        offset = transform_coords(M_inv, [0, 0, current_z])
        offset = np.squeeze(offset) - offset_offset - coords_min

        coords_max = np.ceil(coords_max).astype(np.int64) + 1

        idx0 = slice(coords_min[0], coords_max[0])
        idx1 = slice(coords_min[1], coords_max[1])
        idx2 = slice(coords_min[2], coords_max[2])

        a = array[idx0, idx1, idx2]

        tr = transform(a, M_inv, temp_shape, offset)
        if tr.size:
            q.put(tr)
        current_z += h

    q.put(None)


def main():
    args = parse_args()

    input_file = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_file = (output_dir / input_file.name).with_suffix('.tiff')

    if output_file.exists():
        logger.warning('Output file {} already exists'.format(output_file))
        if not args.force:
            logger.error('(use -f to force)')
            return

    output_dir.mkdir(parents=True, exist_ok=True)

    infile = InputFile(input_file)
    ashape = np.flipud(np.array(infile.shape))  # X, Y, Z order

    M_inv, final_shape = inv_matrix(
        shape=ashape,
        theta=args.theta,
        direction=args.direction,
        view=args.view,
        z=args.z,
        xy=args.xy
    )

    logger.info('input_shape: {}, output_shape: {}'
                .format(infile.shape, tuple(final_shape)))

    total_byte_size = np.asscalar(np.prod(final_shape) * infile.dtype.itemsize)
    bigtiff = total_byte_size > 2 ** 31 - 1

    logger.info('loading {}'.format(input_file))

    a = infile.whole()

    threads = []

    nthreads = args.nthreads
    if nthreads == -1:
        nthreads = int(os.environ['OMP_NUM_THREADS'])

    if args.jp2ar_enabled:
        p = output_file.with_suffix('.zip')
        logger.info(f'saving JP2000 ZIP archive to {p}, using {nthreads} threads')
        jp2ar_thread = threading.Thread(target=convert_to_jp2ar, kwargs=dict(
            input_data=a, output_dir=None, compression=args.jp2_compression,
            nthreads=nthreads, temp_dir=None, output_file=str(p)))
        jp2ar_thread.start()
        threads.append(jp2ar_thread)

    def worker():
        if args.slices is None:
            t = transform(a.T, M_inv, final_shape)  # X, Y, Z order
            logger.info('saving to {}'.format(output_file))
            tiff.imwrite(output_file, t.T, bigtiff=bigtiff, compression='zlib')
            return

        output_file.unlink(missing_ok=True)  # remove file

        i = 0
        for t in sliced_transform(a, M_inv, final_shape, args.slices):
            i += 1
            logger.info('saving slice {}/{} to {}'.format(i, args.slices, output_file))

            t = t.T  # Z, Y, X order

            # add dummy color axis to trick imsave
            # (otherwise when size of Z is 3, it thinks it's an RGB image)
            t = t[:, np.newaxis, ...]
            tiff.imwrite(output_file, t, append=True, bigtiff=bigtiff, compression='zlib')

    transform_thread = threading.Thread(target=worker)
    transform_thread.start()
    threads.append(transform_thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
