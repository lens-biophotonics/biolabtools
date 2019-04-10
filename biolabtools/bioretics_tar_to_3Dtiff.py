import os
import re
import glob
import queue
import shutil
import tarfile
import logging
import argparse
import tempfile
import threading

import coloredlogs
import imageio
import numpy as np
import skimage.external.tifffile as tiff

from zetastitcher import InputFile


logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converts a tar of pngs produced by Bioretics to a mosaic'
                    'of 3D stacks',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('input_file', type=str)
    parser.add_argument('reference_directory', type=str)
    parser.add_argument('output_directory', type=str)

    parser.add_argument('-c', type=str, default=0, dest='compression',
                        choices=[str(i) for i in range(10)] + ['lzma'],
                        help='compression')

    args = parser.parse_args()

    try:
        args.compression = int(args.compression)
    except ValueError:
        pass

    return args


def main():
    def worker():
        while True:
            got = q.get()

            if got is None:
                return

            start, ofname = got

            a = np.zeros((stack_shape[0], stack_shape[-2], stack_shape[-1]),
                         dtype=np.uint8)
            for j in range(nfrms):
                a[j] = imageio.imread(names[start + j])

            logger.info('Writing {}'.format(ofname))
            tiff.imsave(ofname, a, compress=args.compression)

    args = parse_args()

    tar = tarfile.open(args.input_file)
    tar_outer_dir = tar.firstmember.name

    logger.info('Getting names of files in reference folder...')
    files = glob.glob(os.path.join(args.reference_directory, '*x_*.tiff'))

    d = {}

    for f in files:
        m = re.search('^(\d+)x_.*', os.path.basename(f))
        i = int(m.group(1))
        d[i] = f

    tmproot = '/mnt/ramdisk'
    tmproot = tmproot if os.path.exists(tmproot) else None
    logger.info('Unpacking archive to temporary directory...')
    with tempfile.TemporaryDirectory(dir=tmproot) as td:
        shutil.unpack_archive(args.input_file, td)

        names = sorted(glob.glob(os.path.join(td, tar_outer_dir, '*')))

        stack_shape = InputFile(files[0]).shape
        nfrms = stack_shape[0]
        if len(names) != len(d) * nfrms:
            raise RuntimeError('Different number of items in tar archive ({}) '
                               'and in reference folder {} (nfrms: {})'
                               .format(len(names), len(d), nfrms))

        os.makedirs(args.output_directory, exist_ok=True)
        i = 0
        q = queue.Queue(maxsize=os.cpu_count())
        threads = []

        for _ in range(q.maxsize):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        for _, reference_file in sorted(d.items()):
            output_filename = os.path.join(args.output_directory,
                                           os.path.basename(reference_file))
            q.put((i * nfrms, output_filename))
            i += 1

        for _ in threads:
            q.put(None)

        for t in threads:
            t.join()


if __name__ == '__main__':
    main()
