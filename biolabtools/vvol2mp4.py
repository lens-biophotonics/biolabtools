import os
import sys
import math
import argparse
import subprocess as sp

import psutil
import numpy as np

from stitcher import VirtualFusedVolume


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert a stitched volume to a series of compressed '
                    'video files.',
        epilog='Author: Giacomo Mazzamuto <mazzamuto@lens.unifi.it>\n',
        formatter_class=CustomFormatter)

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

    pix_fmt_in = 'gray'
    pix_fmt_out = 'gray'
    if vfv.nchannels == 3 and vfv.dtype == np.uint8:
        pix_fmt_in = 'rgb24'
        pix_fmt_out = 'yuv444p'
    elif vfv.nchannels == 1 and vfv.dtype == np.uint16:
        pix_fmt_in = 'gray16{}'.format(
            'le' if sys.byteorder == 'little' else 'be')

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
        '-preset', 'medium',
        'output.mp4'
    ]

    os.makedirs(args.output_dir, exist_ok=True)

    for j in range(0, int(math.ceil(vfv.shape[-2] / temp_shape[-2]))):
        for i in range(0, int(math.ceil(vfv.shape[-1] / temp_shape[-1]))):
            y = j * temp_shape[-2]
            x = i * temp_shape[-1]

            output_file = '{:05}_{:05}.mp4'.format(x, y)
            command[-1] = os.path.join(args.output_dir, output_file)

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
