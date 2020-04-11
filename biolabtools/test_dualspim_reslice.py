import unittest
from ddt import ddt, data

import numpy as np

import biolabtools.dualspim_reslice as dsr

test_vectors = [
    [
        {
            'shape': (256, 256, 100),
            'theta': 35,
            'r': 2,
            'direction': 'l',
            'view': 'l',
        },
        (555, 256, 147)
    ],
]


@ddt
class TestDualspimReslice(unittest.TestCase):
    @data(*test_vectors)
    def testOutputShape(self, value):
        params = value[0]
        final_shape = value[1]

        _, ret_shape = dsr.inv_matrix(**params)
        np.testing.assert_array_equal(final_shape, ret_shape)

    def testSlicedTransform(self):
        n = 128
        input_a = np.ones((n, n, 100), np.uint8)
        for i in range(input_a.shape[-1]):
            input_a[..., i] = i

        M_inv, final_shape = dsr.inv_matrix(input_a.shape, 45, 6, 'l', 'l')

        sliced_tr = np.zeros(final_shape, input_a.dtype)
        curr_z = 0
        for t in dsr.sliced_transform(input_a, M_inv, final_shape):
            sliced_tr[..., curr_z:curr_z + t.shape[-1]] = t
            curr_z += t.shape[-1]

        tr = dsr.transform(input_a, M_inv, final_shape)
        np.testing.assert_array_almost_equal(tr, sliced_tr, decimal=0)


if __name__ == '__main__':
    unittest.main()
