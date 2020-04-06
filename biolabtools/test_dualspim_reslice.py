import unittest
from ddt import ddt, data

from biolabtools.dualspim_reslice import inv_matrix

test_vectors = [
    [
        {
            'shape': (2048, 2048, 3005),
            'theta': 45,
            'r': 5.9911
        },
        (26897, 2048, 1447)
    ],
]

@ddt
class TestDualspimReslice(unittest.TestCase):
    @data(*test_vectors)
    def testOutputShape(self, value):
        params = value[0]
        final_shape = value[1]

        _, ret_shape = inv_matrix(**params)
        self.assertEqual(final_shape, ret_shape)


if __name__ == '__main__':
    unittest.main()
