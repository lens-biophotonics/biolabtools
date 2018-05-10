import math
import os.path
import logging
import tarfile
import tempfile
import argparse
import threading

from queue import Queue

import glymur
import coloredlogs

from zetastitcher import InputFile

RAMDISK_PATH = '/mnt/ramdisk'

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an input stack into a TAR of compressed images',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>')

    parser.add_argument('-c', type=int, help='compression')
    parser.add_argument('-n', type=int, default=8,
                        help='number of parallel thread to use')

    parser.add_argument('--temp-dir', type=str, default=None,
                        help='full path to temporary directory to be used')

    parser.add_argument('input_file', type=str, help='input file')
    parser.add_argument('output_dir', type=str, help='output directory')

    args = parser.parse_args()
    return args


def convert_to_jp2ar(input_file, output_dir, compression, nthreads,
                     temp_dir=None):
    def waiter():
        while True:
            thr = thread_q.get()
            if thr is None:
                break
            thr.join()

    def save_file(arr, tar, full_path, compression):
        glymur.Jp2k(full_path, data=arr, cratios=[compression])
        with tar_lock:
            tar.add(full_path, arcname=os.path.split(full_path)[1])
        os.remove(full_path)

    thread_q = Queue(nthreads)

    infile = InputFile(input_file)

    w = threading.Thread(target=waiter)
    w.start()

    dir = temp_dir
    if dir is None and os.path.exists(RAMDISK_PATH):
        dir = RAMDISK_PATH

    outfile = os.path.split(input_file)[1]
    outfile = '{}.tar'.format(os.path.splitext(outfile)[0])
    outfile = os.path.join(output_dir, outfile)
    tf = tarfile.open(outfile, mode='w')

    tar_lock = threading.Lock()
    n_of_digits = math.ceil(math.log10(infile.shape[0]))
    fmt = '{:0' + str(n_of_digits) + '}.jp2'

    with tempfile.TemporaryDirectory(dir=dir) as td:
        for k in range(0, infile.shape[0]):
            fname = fmt.format(k)
            full_path = os.path.join(td, fname)
            a = infile[k]  # read frame

            t = threading.Thread(target=save_file,
                                 args=(a, tf, full_path, compression))
            t.start()
            thread_q.put(t)

        thread_q.put(None)
        w.join()
    tf.close()


def main():
    args = parse_args()
    convert_to_jp2ar(
        input_file=args.input_file,
        output_dir=args.output_dir,
        compression=args.c,
        nthreads=args.n,
        temp_dir=args.temp_dir
    )


if __name__ == '__main__':
    main()
