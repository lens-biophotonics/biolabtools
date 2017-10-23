import os
import sys
import math
import logging
import argparse
import subprocess as sp

import psutil
import coloredlogs
import numpy as np

from stitcher import VirtualFusedVolume


logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', fmt='%(levelname)s [%(name)s]: %(message)s')


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a stitched volume to a series of compressed '
                    'video files.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n',
        formatter_class=CustomFormatter)

    group = parser.add_argument_group(
        'Grayscale options', description='These options apply for 16 bit '
                                         'grayscale images only.')
    group.add_argument('--lshift', type=int, default=None,
                       choices=[0, 1, 2, 3, 4, 5, 6, 7], help='left shift')
    group.add_argument('--bpp', type=int, default=None, choices=[8, 12],
                       help='how many bits per pixel will be taken (msb) '
                            'after saturation and shift')

    parser.add_argument('stitch_file', type=str,
                        help='input file (stitch.yml)')
    parser.add_argument('output_dir', type=str, help='output directory')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    vfv = VirtualFusedVolume(args.stitch_file)

    ram = psutil.virtual_memory().available
    # size in bytes of an xy plane (including channels) (float32)
    temp_shape = list(vfv.shape)
    temp_shape[-1] = 1024
    temp_shape[-2] = 1024
    xy_size = np.asscalar(np.prod(temp_shape[1::]) * 4)
    n_frames_in_ram = int(ram / xy_size / 1.5)

    pix_fmt_in = None
    pix_fmt_out = None
    if vfv.nchannels == 1 and vfv.dtype == np.uint16:
        pix_fmt_in = 'gray16{}'.format(
            'le' if sys.byteorder == 'little' else 'be')
        if args.bpp == 12:
            pix_fmt_out = 'gray12le'
        elif args.bpp == 8:
            pix_fmt_out = 'gray'
        elif args.bpp is None:
            sys.exit('Error: option --bpp required for 16 bit images.')
        else:
            raise ValueError('Invalid bpp: {}'.format(args.bpp))
    if vfv.dtype == np.uint8:
        if args.bpp is not None or args.lshift is not None:
            logger.warning('Ignoring bpp and lshift arguments for 8bpp or RGB '
                           'images')
        args.bpp = None
        args.lshift = None
        if vfv.nchannels == 3:
            pix_fmt_in = 'rgb24'
            pix_fmt_out = 'yuv444p'
        elif vfv.nchannels == 1:
            pix_fmt_in = 'gray'
            pix_fmt_out = 'gray'
    if pix_fmt_in is None or pix_fmt_out is None:
        raise ValueError('Invalid nchannels and pix_fmt combination')

    command = [
        'ffmpeg',
        '-y',  # overwrite output file
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', '1024x1024',  # frame size
        '-pix_fmt', pix_fmt_in,
        '-r', '30',  # frames per second
        '-i', '-',  # read input from stdin
        '-an',  # no audio
        '-c:v', 'libx265',
        '-pix_fmt', pix_fmt_out,
        '-crf', '23',
        'output.mp4'
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    jmax = int(math.ceil(vfv.shape[-2] / temp_shape[-2]))
    imax = int(math.ceil(vfv.shape[-1] / temp_shape[-1]))

    for j in range(0, jmax):
        for i in range(0, imax):
            y = j * temp_shape[-2]
            x = i * temp_shape[-1]

            output_file = '{:05}_{:05}.mp4'.format(x, y)
            command[-1] = os.path.join(args.output_dir, output_file)

            logger.info('Progress: {:.2f}%, output_file: {}'.format(
                (j * imax + i) / (jmax * imax) * 100, output_file))
            pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)

            z = 0
            while z < vfv.shape[0]:
                z_stop = z + n_frames_in_ram
                if z_stop > vfv.shape[0]:
                    z_stop = vfv.shape[0]
                idx = (
                    slice(z, z + n_frames_in_ram),
                    Ellipsis,
                    slice(y, y + temp_shape[-2]),
                    slice(x, x + temp_shape[-1])
                )
                a = vfv[idx]

                if args.lshift:
                    # gray16 to gray12 takes the 12 most significant bits
                    # => we need to saturate and perform a left shift
                    nbytes = np.dtype(vfv.dtype).itemsize
                    cond = a > 2**(args.bpp + args.lshift) - 1
                    a[cond] = 2**(nbytes * 8) - 1
                    a = a << (nbytes * 8 - args.bpp - args.lshift)
                if np.prod(temp_shape[-2::]) != np.prod(a.shape[-2::]):
                    padding = [(0, 0) for _ in temp_shape]
                    padding[-1] = (0, temp_shape[-1] - a.shape[-1])
                    padding[-2] = (0, temp_shape[-2] - a.shape[-2])
                    a = np.pad(a, padding, 'constant')

                if vfv.nchannels > 1:
                    a = np.moveaxis(a, -3, -1)  # channel dimension to last pos

                pipe.stdin.write(a.tostring())
                z = z_stop

            pipe.communicate()


if __name__ == '__main__':
    main()
