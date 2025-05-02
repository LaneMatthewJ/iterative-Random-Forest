# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _tree.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

from ._splitter cimport Splitter
from ._splitter cimport SplitRecord

cdef struct Node:
    SIZE_t left_child
    SIZE_t right_child
    SIZE_t feature
    DOUBLE_t threshold
    DOUBLE_t impurity
    SIZE_t n_node_samples
    DOUBLE_t weighted_n_node_samples

cdef class Tree:
    cdef public SIZE_t n_features
    cdef SIZE_t* n_classes
    cdef public SIZE_t n_outputs
    cdef public SIZE_t max_n_classes

    cdef public SIZE_t max_depth
    cdef public SIZE_t node_count
    cdef public SIZE_t capacity
    cdef Node* nodes
    cdef double* value
    cdef SIZE_t value_stride

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_samples) except -1 nogil

    cdef int _resize(self, SIZE_t capacity) except -1 nogil
    cdef int _resize_c(self, SIZE_t capacity=*) except -1 nogil

    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)

    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray apply(self, object X)
    cdef np.ndarray _apply_dense(self, object X)
    cdef np.ndarray _apply_sparse_csr(self, object X)

    cpdef object decision_path(self, object X)
    cdef object _decision_path_dense(self, object X)
    cdef object _decision_path_sparse_csr(self, object X)

    cpdef compute_feature_importances(self, normalize=*)


# =============================================================================
# Tree builder
# =============================================================================

cdef class TreeBuilder:
    cdef Splitter splitter

    cdef SIZE_t min_samples_split
    cdef SIZE_t min_samples_leaf
    cdef double min_weight_leaf
    cdef SIZE_t max_depth
    cdef double min_impurity_split
    cdef double min_impurity_decrease

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=*,
                np.ndarray feature_weight=*,
                np.ndarray X_idx_sorted=*)

    cdef tuple _check_input(self, object X, np.ndarray y,
                           np.ndarray sample_weight,
                           np.ndarray feature_weight)