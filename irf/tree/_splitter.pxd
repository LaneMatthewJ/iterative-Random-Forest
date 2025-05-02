# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

# See _splitter.pyx for details.

import numpy as np
cimport numpy as np

from ._criterion cimport Criterion

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef struct SplitRecord:
    SIZE_t feature
    SIZE_t pos
    double threshold
    double improvement
    double impurity_left
    double impurity_right

cdef class Splitter:
    cdef public Criterion criterion
    cdef public SIZE_t max_features
    cdef public SIZE_t min_samples_leaf
    cdef public double min_weight_leaf

    cdef object random_state
    cdef UINT32_t rand_r_state

    cdef SIZE_t* samples
    cdef SIZE_t n_samples
    cdef double weighted_n_samples
    cdef SIZE_t* features
    cdef SIZE_t* constant_features
    cdef SIZE_t n_features
    cdef DTYPE_t* feature_values

    cdef SIZE_t start
    cdef SIZE_t end

    cdef bint presort

    cdef DOUBLE_t* y
    cdef SIZE_t y_stride
    cdef DOUBLE_t* sample_weight
    cdef DOUBLE_t* feature_weight

    # Methods
    cdef int init(self, object X, np.ndarray y,
                  DOUBLE_t* sample_weight,
                  DOUBLE_t* feature_weight,
                  np.ndarray X_idx_sorted=*) except -1

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) except -1 nogil

    cdef int node_split(self,
                        double impurity,
                        SplitRecord* split,
                        SIZE_t* n_constant_features) except -1 nogil

    cdef void node_value(self, double* dest) nogil

    cdef double node_impurity(self) nogil