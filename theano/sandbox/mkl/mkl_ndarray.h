#ifndef _MKL_NDARRAY_H_
#define _MKL_NDARRAY_H_

#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdint.h>
#include "mkl_dnn.h"

#ifndef SIZE_MAX
#define SIZE_MAX ((size_t) - 1)
#endif

#ifndef Py_TYPE
#define Py_TYPE(o) ((o)->ob_type)
#endif

char* MKL_TYPE[] = {"", "", "", "int16", "", "int32", "", "int64",
                      "", "", "", "float32", "float64", ""};

/**
 * struct : wrapper for MKL internal data and layout
 *
 * This is a Python type.
 *
 */
typedef struct __MKLNdarray__
{
    PyObject_HEAD

    PyObject * base;

    /* Type-specific fields go here. */
    int nd;                 // the number of dimensions of the tensor
    int dtype;              // ...
    size_t mkldata_size;    // the number of bytes allocated for mkldata
    size_t* user_structure; // user layout: [size0, size1, ..., stride0, stride1, ...]
    dnnLayout_t mkl_layout;
    void* mkl_data;
}MKLNdarray;


__attribute__((visibility ("default"))) int MKLNdarray_Check(const PyObject* ob);
__attribute__((visibility ("default"))) PyObject* MKLNdarray_New(int nd, int typenum);
__attribute__((visibility ("default"))) int MKLNdarray_CopyFromArray(MKLNdarray* self, PyArrayObject* obj);
__attribute__((visibility ("default"))) int MKLNdarray_set_structure(MKLNdarray* self, int nd, size_t* dims);
__attribute__((visibility ("default"))) PyObject* MKLNdarray_CreateArrayObj(MKLNdarray* self);
#endif
