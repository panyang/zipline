import numpy as np

from zipline.lib.stringarray import LabelArray
from zipline.testing import check_arrays, parameter_space, ZiplineTestCase

def rotN(l, N):
    """
    Rotate a list of elements.

    Pulls N elements off the end of the list and appends them to the front.

    >>> rotN(['a', 'b', 'c', 'd'], 2)
    ['c', 'd', 'a', 'b']
    >>> rotN(['a', 'b', 'c', 'd'], 3)
    ['d', 'a', 'b', 'c']
    """
    assert len(l) >= N, "Can't rotate list by longer than its length."
    return l[N:] + l[:N]


class LabelArrayTestCase(ZiplineTestCase):

    @classmethod
    def init_class_fixtures(cls):
        super(LabelArrayTestCase, cls).init_class_fixtures()
        rand = np.random.RandomState(5)

        cls.rowvalues = row = ['', 'aa', 'ab', 'ba', 'ba', 'aa', 'z', 'ab', 'z']
        cls.strs = np.array([rotN(row, i) for i in range(3)])

    def test_fail_on_direct_construction(self):
        # See http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray  # noqa

        with self.assertRaises(TypeError) as e:
            np.ndarray.__new__(LabelArray, (5, 5))

        self.assertEqual(
            str(e.exception),
            "Direct construction of LabelArrays is not supported."
        )

    @parameter_space(
        s=['', 'a', 'z', 'aa', 'not in the array'],
        shape=[(9, 3), (3, 9), (3, 3, 3)],
    )
    def test_compare_to_str(self, s, shape):
        strs = self.strs.reshape(shape)
        arr = LabelArray(strs)
        check_arrays(strs == s, arr == s)

    @parameter_space(
        slice_=[
            0, 1, -1,
            slice(None),
            slice(0, 0),
            slice(0, 3),
            slice(1, 4),
            slice(0),
            slice(None, 1),
            slice(0, 4, 2),
            (slice(None), 1),
            (slice(None), slice(None)),
            (slice(None), slice(1, 2)),
        ]
    )
    def test_slicing_preserves_type_and_categories(self, slice_):
        arr = LabelArray(self.strs.reshape((9, 3)))
        sliced = arr[slice_]
        self.assertIsInstance(sliced, LabelArray)
        self.assertEqual(sliced.categories, arr.categories)

    def test_infer_categories(self):
        arr1d = LabelArray(self.strs)
        self.assertEqual(arr1d.shape, self.strs.shape)

        categories1d = arr1d.categories

        # We should have an entry for each unique row value.
        unique_rowvalues = set(self.rowvalues)

        self.assertEqual(set(categories1d), unique_rowvalues)
        # Each entry in the array should be an index into the categories array.
        self.assertEqual(
            set(arr1d.view(type=ndarray)),
            set(range(len(unique_rowvalues)))
        )

        for idx, value in enumerate(arr1d.categories):
            check_arrays(
                self.strs == value,
                arr1d.view(type=np.ndarray) == idx,
            )

        for shape in (9, 3), (3, 9), (3, 3, 3):
            arr2d = LabelArray(self.strs.reshape(shape))

            self.assertEqual(arr2d.shape, shape)
            self.assertEqual(categories1d, arr2d.categories)

            for idx, value in enumerate(arr2d.categories):
                check_arrays(
                    self.strs == value,
                    arr2d.view(type=np.ndarray) == idx,
                )
