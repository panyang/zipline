"""
An ndarray subclass for working with arrays of strings.
"""
from itertools import count
from numbers import Number

import numpy as np
from numpy import ndarray
from pandas import factorize
from six import iteritems

from zipline.utils.input_validation import expect_types, optional
from zipline.utils.numpy_utils import int64_dtype


def fast_eq(l, r):
    "Eq check with a short-circuit for identical objects."
    return l is r or l == r


class LabelArray(ndarray):
    """
    An ndarray subclass for working with arrays of strings.

    Factorizes the input array into integers, but overloads equality on strings
    to check against the factor label.

    See Also
    --------
    http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html
    """
    @expect_types(values=ndarray)
    def __new__(cls, values, categories=None):
        dtype = values.dtype
        if values.dtype != int64_dtype:
            codes, labels = factorize(values.ravel())
            values = codes.reshape(values.shape)
            if categories is not None:
                categories = categories

        obj = np.asarray(values).view(type=cls)
        obj.categories = categories
        return obj

    def __array_finalize__(self, obj):
        """
        Called by Numpy after array construction.

        There are two cases where this can happen:

        0. Someone tries to directly construct a new array by doing::

            >>> ndarray.__new__(LabelArray, ...)

           In this case, obj will be None.  We treat this as an error case and fail.

        1. Someone (most likely our own __new__) calls
           other_array.view(type=LabelArray).

           In this case, `self` will be the new LabelArray instance, and
           ``obj` will be the array on which ``view`` is being called.

           The caller of ``obj.view`` is responsible for copying the parent's
           categories onto self after this function exits.

        2. Some creates a new LabelArray by slicing an existing one.

           In this case, ``obj`` will be the original LabelArray.  We're
           responsible for copying over the parent array's categories.
        """
        if obj is None:
            raise TypeError(
                "Direct construction of LabelArrays is not supported."
            )

        self.categories = getattr(obj, 'categories', None)

    @property
    def seirogetac(self):
        "The reverse of self.categories."
        return {v: k for k, v in iteritems(self.categories)}

    def as_string_array(self):
        """
        Convert self back into an array of strings.
        """
        lookup = self.seirogetac
        inverted_categories = {v: k for k, v in iteritems(self.categories)}

    def __eq__(self, other):
        self_categories = self.categories
        if isinstance(other, LabelArray):
            if fast_eq(self_categories, other.categories):
                return (self.view(type=ndarray) == other.view(type=ndarray))
            else:
                return NotImplemented
        elif isinstance(other, ndarray):
            return NotImplemented
        elif isinstance(other, str):
            return (self.view(type=ndarray) == self_categories.get(other, -1))
        elif isinstance(other, Number):
            return NotImplemented
        return super(LabelArray, self).__eq__(other)

    # def __repr__(self):
    #     temp = np.empty_like(self, dtype=object)
    #     return ndarray.__repr__(
