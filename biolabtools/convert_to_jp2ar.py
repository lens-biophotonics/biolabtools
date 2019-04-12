import math
import os.path
import logging
import zipfile
import tempfile
import argparse
import threading

from queue import Queue

import glymur
import coloredlogs

from zetastitcher import InputFile

RAMDISK_PATH = '/mnt/ramdisk'

logger = logging.getLogger(__name__)
coloredlogs.install(level='INFO', fmt='%(levelname)s [%(name)s]: %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert an input stack into a ZIP of compressed images',
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


def convert_to_jp2ar(input_data, output_dir, compression, nthreads,
                     temp_dir=None, output_file=None):
    """

    Parameters
    ----------
    input_data : str (filename) or object implementing shape and __getitem__
    output_dir : str
        can be None if output_file is specified
    compression
    nthreads
    temp_dir
    output_file : str
        must be specified if input_file is not a string
    """
    def waiter():
        while True:
            thr = thread_q.get()
            if thr is None:
                break
            thr.join()

    def save_file(arr, zf, full_path, compression):
        glymur.Jp2k(full_path, data=arr, cratios=[compression])
        with tar_lock:
            zf.write(full_path, arcname=os.path.split(full_path)[1])
        os.remove(full_path)

    thread_q = Queue(nthreads)

    if type(input_data) is str:
        out_file_name = os.path.split(input_data)[1]
        out_file_name = '{}.zip'.format(os.path.splitext(out_file_name)[0])
        out_file_name = os.path.join(output_dir, out_file_name)
        os.makedirs(output_dir, exist_ok=True)
        input_data = InputFile(input_data)
    else:
        out_file_name = output_file

    w = threading.Thread(target=waiter)
    w.start()

    dir = temp_dir
    if dir is None and os.path.exists(RAMDISK_PATH):
        dir = RAMDISK_PATH

    zf = zipfile.ZipFile(out_file_name, mode='w',
                         compression=zipfile.ZIP_STORED)

    tar_lock = threading.Lock()
    n_of_digits = math.ceil(math.log10(input_data.shape[0]))
    fmt = '{:0' + str(n_of_digits) + '}.jp2'

    with tempfile.TemporaryDirectory(dir=dir) as td:
        for k in range(0, input_data.shape[0]):
            fname = fmt.format(k)
            full_path = os.path.join(td, fname)
            a = input_data[k]  # read frame
            if k % 100 == 0:
                logger.info('JPEG2000 Progress: {:.2f}%'.format(
                    k / input_data.shape[0] * 100))

            t = threading.Thread(target=save_file,
                                 args=(a, zf, full_path, compression))
            t.start()
            thread_q.put(t)

        thread_q.put(None)
        w.join()
    zf.close()


def main():
    args = parse_args()
    convert_to_jp2ar(
        input_data=args.input_file,
        output_dir=args.output_dir,
        compression=args.c,
        nthreads=args.n,
        temp_dir=args.temp_dir
    )


if __name__ == '__main__':
    main()
