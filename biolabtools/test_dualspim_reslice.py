import math
import unittest
from ddt import ddt, data

import numpy as np

import biolabtools.dualspim_reslice as dsr

test_vectors = [
    [
        {
            'shape': (256, 256, 100),
            'theta': 33,
            'z': 2,
            'direction': 'l',
            'view': 'l',
        },
        (576, 256, 139)
    ],
    [
        {
            'shape': (256, 256, 100),
            'theta': 45,
            'z': 2,
            'direction': 'l',
            'view': 'l',
        },
        (460, 256, 181)
    ],
]


@ddt
class TestDualspimReslice(unittest.TestCase):
    @data(*test_vectors)
    def testOutputShape(self, value):
        params = value[0]
        theta = params['theta'] * np.pi / 180
        shape = np.array(params['shape'])
        edge = shape - 1
        z = params['z']
        expected_shape = np.array([
            z * edge[2] / math.sin(theta) + edge[0] * math.cos(theta),
            shape[1],
            shape[0] * math.sin(theta)
        ]).astype(np.int64)

        for view in 'lr':
            params['view'] = view
            _, ret_shape = dsr.inv_matrix(**params)
            np.testing.assert_almost_equal(value[1], expected_shape, decimal=0)
            np.testing.assert_equal(value[1], ret_shape)

    @data(*test_vectors)
    def testSlicedTransform(self, value):
        params = value[0]
        input_a = np.ones(params['shape'], np.uint8)
        for i in range(input_a.shape[-1]):
            input_a[..., i] = i
        input_a[..., 0] = 1

        for view in 'lr':
            params['view'] = view
            M_inv, output_shape = dsr.inv_matrix(**params)

            sliced_tr = np.zeros(output_shape, input_a.dtype)
            curr_z = 0
            for t in dsr.sliced_transform(input_a, M_inv, output_shape):
                sliced_tr[..., curr_z:curr_z + t.shape[-1]] = t
                curr_z += t.shape[-1]

            tr = dsr.transform(input_a, M_inv, output_shape)

            np.testing.assert_array_almost_equal(tr, sliced_tr, decimal=0)

            # check corners
            if view == 'r':
                self.assertTrue(np.all(tr[0, :, 0]))
                self.assertTrue(np.all(tr[-1, :, -1]))
            elif view == 'l':
                self.assertTrue(np.all(tr[-1, :, 0]))
                self.assertTrue(np.all(tr[0, :, -1]))


if __name__ == '__main__':
    unittest.main()
