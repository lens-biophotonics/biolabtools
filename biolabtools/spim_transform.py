import os
import time
import logging
import argparse
import threading
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
        description='Apply affine transform to SPIM images',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>',
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument('-z', type=float, default=1, help='zoom factor along z')
    parser.add_argument('-xy', type=float, default=1, help='zoom factor along xy')
    parser.add_argument('-f', '--force', action='store_true',
                        help="don't stop if output file exists")
    parser.add_argument('-o', '--output_dir', type=str)
    parser.add_argument('-m', '--initial_matrix', type=str, help='Path to initial matrix (.csv)')

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


def inv_matrix(M_init, z=1, xy=1):
    """
    Compute inverse coordinate transformation matrix.

    Parameters
    ----------
    M_init : :class:`numpy.ndarray`
            initial matrix
    z : float
        zoom factor along z

    xy : float
        zoom factor along xy

    Returns
    -------
    M_inv : :class:`numpy.ndarray`
            The inverse coordinate transformation inv_matrix, mapping output
            coordinates to input coordinates. To be used in
            :func:`scipy.ndimage.affine_transform`
    """
    # zoom matrix
    Z = [xy, xy, z, 1] * np.eye(4)

    M = np.linalg.multi_dot([Z, M_init])
    M_inv = np.linalg.inv(M)

    return M_inv


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

    M_init = np.eye(4)
    if args.initial_matrix:
        M_init = np.loadtxt(args.initial_matrix, delimiter=',')

    output_dir.mkdir(parents=True, exist_ok=True)

    infile = InputFile(input_file)

    # zoom matrix
    Z = [args.xy, args.xy, args.z, 1] * np.eye(4)

    M = np.linalg.multi_dot([Z, M_init])
    M_inv = np.linalg.inv(M)

    final_shape = np.array(infile.shape) * np.array([args.z, args.xy, args.xy])
    final_shape = final_shape.astype(np.uint64)

    logger.info(f'input_shape: {infile.shape}, output_shape: {tuple(final_shape)}')

    total_byte_size = (np.prod(final_shape) * infile.dtype.itemsize).item()
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
        t = transform(a.T, M_inv, np.flipud(final_shape))  # X, Y, Z order
        logger.info('saving to {}'.format(output_file))
        tiff.imwrite(output_file, t.T, bigtiff=bigtiff, compression='zlib')

    transform_thread = threading.Thread(target=worker)
    transform_thread.start()
    threads.append(transform_thread)

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
