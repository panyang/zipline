"""
An ndarray subclass for working with arrays of strings.
"""
from functools import partial
from numbers import Number
from operator import eq, ne

import numpy as np
from numpy import ndarray
from pandas import factorize

from zipline.utils.preprocess import preprocess
from zipline.utils.input_validation import coerce, expect_kinds, expect_types
from zipline.utils.numpy_utils import (
    is_object,
    is_string,
    int64_dtype,
)


def fast_eq(l, r):
    "Eq check with a short-circuit for identical objects."
    return l is r or (l == r).all()


class CategoryMismatch(ValueError):
    """
    Error raised on attempt to perform operations between LabelArrays with
    mismatched category arrays.
    """
    def __init__(self, left, right):
        (mismatches,) = np.where(left != right)
        assert len(mismatches), "Not actually a mismatch!"
        super(CategoryMismatch, self).__init__(
            "LabelArray categories don't match:\n"
            "Mismatched Indices: {mismatches}\n"
            "Left: {left}\n"
            "Right: {right}".format(
                mismatches=mismatches,
                left=left[mismatches],
                right=right[mismatches],
            )
        )


class LabelArray(ndarray):
    """
    An ndarray subclass for working with arrays of strings.

    Factorizes the input array into integers, but overloads equality on strings
    to check against the factor label.

    See Also
    --------
    http://docs.scipy.org/doc/numpy-1.10.0/user/basics.subclassing.html
    """
    @preprocess(values=coerce(list, partial(np.asarray, dtype=object)))
    @expect_types(values=np.ndarray)
    @expect_kinds(values=("S", "O"))
    def __new__(cls, values):

        if is_string(values):
            values = values.astype(object)

        codes, categories = factorize(values.ravel(), sort=True)
        categories.setflags(write=False)

        ret = codes.reshape(values.shape).view(type=cls)
        ret._categories = categories
        ret._reverse_categories = (
            dict(zip(categories, np.arange(len(categories))))
        )
        return ret

    @property
    def categories(self):
        # This is a property because it should be immutable.
        return self._categories

    @property
    def reverse_categories(self):
        # This is a property because it should be immutable.
        return self._reverse_categories

    def __array_finalize__(self, obj):
        """
        Called by Numpy after array construction.

        There are three cases where this can happen:

        1. Someone tries to directly construct a new array by doing::

            >>> ndarray.__new__(LabelArray, ...)

           In this case, obj will be None.  We treat this as an error case and
           fail.

        2. Someone (most likely our own __new__) calls
           other_array.view(type=LabelArray).

           In this case, `self` will be the new LabelArray instance, and
           ``obj` will be the array on which ``view`` is being called.

           The caller of ``obj.view`` is responsible for copying setting
           category metadata on ``self`` after we exit.

        3. Someone creates a new LabelArray by slicing an existing one.

           In this case, ``obj`` will be the original LabelArray.  We're
           responsible for copying over the parent array's category metadata.
        """
        if obj is None:
            raise TypeError(
                "Direct construction of LabelArrays is not supported."
            )
        if self.dtype != int64_dtype:
            raise TypeError("Can't coerce LabelArrays to other dtypes.")

        # See docstring for an explanation of when these will or will not be
        # set.
        self._categories = getattr(obj, 'categories', None)
        self._reverse_categories = getattr(obj, 'reverse_categories', None)

    def as_int_array(self):
        """
        Convert self into a regular ndarray of ints.

        This is an O(1) operation, and does not copy the underlying data.
        """
        return self.view(type=ndarray)

    def as_string_array(self):
        """
        Convert self back into an array of strings.
        """
        return self.categories[self]

    def __setitem__(self, indexer, value):
        self_categories = self.categories
        if isinstance(value, LabelArray):
            value_categories = value.categories
            if fast_eq(self_categories, value_categories):
                return super(LabelArray, self).__setitem__(indexer, value)
            else:
                raise CategoryMismatch(self_categories, value_categories)
        elif isinstance(value, str):
            value_code = self.reverse_categories.get(value, None)
            if value_code is None:
                raise ValueError("%r is not in LabelArray categories." % value)
            return super(LabelArray, self).__setitem__(indexer, value_code)
        else:
            raise NotImplementedError(
                "Setting into a LabelArray with a value of "
                "type {type} is not yet supported.".format(
                    type=type(value).__name__,
                ),
            )

    def _make_equality_check(op):
        def method(self, other):
            self_categories = self.categories
            if isinstance(other, LabelArray):
                other_categories = other.categories
                if fast_eq(self_categories, other_categories):
                    return op(self.as_int_array(), other.as_int_array())
                else:
                    raise CategoryMismatch(self_categories, other_categories)
            elif isinstance(other, ndarray):
                return op(self.as_string_array(), other)
            elif isinstance(other, str):
                i = self._reverse_categories.get(other, -1)
                if not i:  # Requested string isn't in our categories.
                    return np.full_like(self, False, dtype=bool)
                return op(self.as_int_array(), i)
            elif isinstance(other, Number):
                return NotImplemented
            return super(LabelArray, self).__eq__(other)
        return method

    __eq__ = _make_equality_check(eq)
    __ne__ = _make_equality_check(ne)
    del _make_equality_check

    def __repr__(self):
        repr_lines = repr(self.as_string_array()).splitlines()
        repr_lines[0] = repr_lines[0].replace('array(', 'LabelArray(', 1)
        repr_lines[-1] = repr_lines[-1].rsplit(',', 1)[0] + ')'
        # The extra spaces here account for the difference in length between
        # 'array(' and 'LabelArray('.
        return '\n     '.join(repr_lines)
