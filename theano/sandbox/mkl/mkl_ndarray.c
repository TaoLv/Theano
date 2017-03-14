#include <Python.h>
#include <structmember.h>
#include "theano_mod_helper.h"
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include "mkl_ndarray.h"


int MKLNdarray_Check(const PyObject* ob);
PyObject* MKLNdarray_New(int nd, int typenum);
int MKLNdarray_CopyFromArray(MKLNdarray* self, PyArrayObject* obj);


static int
MKLNdarray_uninit(MKLNdarray* self)
{
    printf("MKLNdarray_uninit %p \n", self);
    int rval = 0;
    if (self->mkl_data)
    {
        if (self->dtype == 12)
            rval = dnnReleaseBuffer_F64(self->mkl_data);
        else
            rval = dnnReleaseBuffer_F32(self->mkl_data);

        if (rval != 0)
        {
            printf("MKLNdarray_uninit: fail to release mkl_data \n");
        }
        self->mkl_data = NULL;
    }
    self->mkldata_size = 0;

    if (self->mkl_layout)
    {
        if (self->dtype == 12)
            rval = dnnLayoutDelete_F64(self->mkl_layout);
        else
            rval = dnnLayoutDelete_F32(self->mkl_layout);

        if (rval != 0)
        {
            printf("MKLNdarray_uninit: fail to release mkl_layout \n");
        }
        self->mkl_layout = NULL;
    }

    if (self->user_structure)
    {
        free(self->user_structure);
        self->user_structure = NULL;
    }

    self->nd = -1;
    self->dtype = 11;

    Py_XDECREF(self->base);
    self->base = NULL;

    return rval;
}


/* type:tp_dealloc
 * This function will be called by Py_DECREF when object's reference count is reduced to zero.
 * DON'T call this function directly.
 */
static void
MKLNdarray_dealloc(MKLNdarray* self)
{
    printf("MKLNdarray_dealloc\n");
    if (Py_REFCNT(self) > 1)
    {
        printf("WARNING: MKLNdarray_dealloc called when there is still active reference to it.\n");
    }

    MKLNdarray_uninit(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}


/* type:tp_new
 * This function is used to create an instance of object.
 * Be first called when do a = MKLNdarray() in python code.
 */
static PyObject *
MKLNdarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    MKLNdarray* self = NULL;
    printf("MKLNdarray_new\n");
    self = (MKLNdarray*)(type->tp_alloc(type, 0));
    if (self != NULL)
    {
        self->base = NULL;
        self->nd = -1;
        self->dtype = 11;
        self->user_structure = NULL;
        self->mkl_data = NULL;
        self->mkl_layout = NULL;
        self->mkldata_size = 0;
    }
    return (PyObject*)self;
}



/* type:tp_init
 * Initialize an instance. __init__().
 */
static int
MKLNdarray_init(MKLNdarray* self, PyObject* args, PyObject* kwds)
{
    printf("MKLNdarray_init\n");
    PyObject* arr = NULL;

    if (!PyArg_ParseTuple(args, "O", &arr))
        return -1;

    if (!PyArray_Check(arr))
    {
        PyErr_SetString(PyExc_TypeError, "PyArray arg required");
        return -1;
    }
    int rval = MKLNdarray_CopyFromArray(self, (PyArrayObject*)arr);
    return rval;
}




/* type:tp_repr
 * Return a string or a unicode object. repr().
 */
PyObject * MKLNdarray_repr(PyObject* self)
{
    printf("MKLNdarray_repr \n");
    MKLNdarray* object = (MKLNdarray*)self;
    char cstr[64]; // TODO: Is it enough?
    sprintf(cstr, "ndim=%d, dtype=%s", object->nd, MKL_TYPE[object->dtype]);
    PyObject * out = PyString_FromFormat("%s%s%s", "MKLNdarray(", cstr, ")");
    // Py_DECREF(object);
#if PY_MAJOR_VERSION >= 3
    PyObject* out2 = PyObject_Str(out);
    Py_DECREF(out);
    return out2;
#endif
    return out;
}


const size_t*
MKLNdarray_USER_DIMS(const MKLNdarray * self)
{
    return self->user_structure;
}

const size_t*
MKLNdarray_USER_STRIDES(const MKLNdarray * self)
{
	return self->user_structure + self->nd;
}


static int
MKLNdarray_allocate_mkl_buffer(MKLNdarray* self)
{
    if (self->nd <= 0)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't create mkl dnn layout and allocate buffer for a %d dimension MKLNdarray",
                     self->nd);
        return -1;
    }
    size_t ndim = self->nd;

    if (self->mkl_layout || self->mkl_data)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "MKL layout and buffer have been allocated for %p \n", self);
        return -1;
    }

    if (self->dtype == 12)
    {
        int status = dnnLayoutCreate_F64(&(self->mkl_layout),
                                         ndim,
                                         self->user_structure,
                                         self->user_structure + self->nd);
        if (0 != status || NULL == self->mkl_layout)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnLayoutCreate_F64 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F64(&(self->mkl_data), self->mkl_layout);
        if (0 != status || NULL == self->mkl_data)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnAllocateBuffer_F64 failed: %d",
                         status);
            return -1;
        }
        self->mkldata_size = dnnLayoutGetMemorySize_F32(self->mkl_layout);
    }
    else
    {
        int status = dnnLayoutCreate_F32(&(self->mkl_layout),
                                                 ndim,
                                                 self->user_structure,
                                                 self->user_structure + self->nd);
        if (0 != status || NULL == self->mkl_layout)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnLayoutCreate_F32 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F32(&(self->mkl_data), self->mkl_layout);
        if (0 != status || NULL == self->mkl_data)
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Call dnnAllocateBuffer_F32 failed: %d",
                         status);
            return -1;
        }
        self->mkldata_size = dnnLayoutGetMemorySize_F64(self->mkl_layout);
    }
    return 0;
}


int MKLNdarray_set_structure(MKLNdarray* self, int nd, size_t* dims)
{
    assert (self->nd == nd);
    if (self->user_structure == NULL)
    {
        self->user_structure = (size_t*)malloc(2 * nd * sizeof (size_t));
    }

    if (self->user_structure == NULL)
    {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate mkl_structure");
        return -1;
    }

    self->user_structure[0] = dims[0];
    self->user_structure[nd] = 1;
    for (int i = 1; i < nd; i++)
    {
        self->user_structure[i] = dims[i];
        self->user_structure[i + nd] = self->user_structure[i + nd - 1] * dims[i - 1];
    }

    return 0;
}

int MKLNdarray_CopyFromArray(MKLNdarray* self, PyArrayObject* obj)
{
    int ndim = PyArray_NDIM(obj);
    npy_intp* d = PyArray_DIMS(obj);
    int typenum = PyArray_TYPE(obj);

    if (typenum != 11 && typenum != 12)
    {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_CopyFromArray: can only copy from float/double arrays");
        return -1;
    }

    self->dtype = typenum;
    if (ndim < 0 || d == NULL)
    {
        return -1;
    }
    self->nd = ndim;
    size_t* dims = (size_t*)malloc(ndim * sizeof(size_t));
    size_t user_size = 1;
    for (int i = 0; i < ndim; i++)
    {
        dims[i] = (size_t)d[i];
        user_size *= dims[i];
    }

    int err = MKLNdarray_set_structure(self, ndim, dims);
    if (err < 0)
    {
        free (dims);
        return err;
    }
    free (dims);

    // prepare user layout and mkl buffer
    err = MKLNdarray_allocate_mkl_buffer(self);
    if (err < 0)
        return err;

    // copy data to mkl buffer
    size_t element_size = (size_t)PyArray_ITEMSIZE(obj);
    assert (user_size * element_size <= self->mkldata_size);
    memcpy((void*)self->mkl_data, (void*)PyArray_DATA(obj), user_size * element_size);
    return 0;
}



/*
 * Create a MKLNdarray object with dims and set all elements zero.
 */
PyObject* MKLNdarray_ZEROS(int n, size_t* dims, int typenum)
{
    size_t total_elements = 1;
    for (int i = 0; i < n; i++)
    {
        if (dims[i] != 0 && total_elements > (SIZE_MAX / dims[i]))
        {
            PyErr_Format(PyExc_RuntimeError,
                         "Can't store in size_t for the bytes requested %llu * %llu",
                         (unsigned long long)total_elements,
                         (unsigned long long)dims[i]);
            return NULL;
        }
        total_elements *= dims[i];
    }

    // total_elements now contains the size of the array, in reals
    size_t max = 0;
    if (typenum == 12)
        max = SIZE_MAX / sizeof (double);
    else
        max = SIZE_MAX / sizeof (float);
    if (total_elements > max)
    {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't store in size_t for the bytes requested %llu",
                     (unsigned long long)total_elements);
        return NULL;
    }

    size_t total_size = 0;
    if (typenum == 12)
        total_size = total_elements * sizeof (double);
    else
        total_size = total_elements * sizeof (float);

    MKLNdarray* rval = (MKLNdarray*)MKLNdarray_New(n, typenum);
    if (!rval)
    {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_ZEROS: call to New failed");
        return NULL;
    }

    if (MKLNdarray_set_structure(rval, n, dims))
    {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_ZEROS: syncing structure to mkl failed.");
        Py_DECREF(rval);
        return NULL;
    }

    if (MKLNdarray_allocate_mkl_buffer(rval))
    {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarrya_ZEROS: allocation failed.");
        Py_DECREF(rval);
        return NULL;
    }
    // Fill with zeros
    memset(rval->mkl_data, 0, total_size);

    return (PyObject*)rval;
}



static PyObject*
MKLNdarray_get_shape(MKLNdarray * self, void * closure)
{
    printf("MKLNdarray_get_shape: ndim = %d \n", self->nd);
    if (self->nd < 0)
    {
        PyErr_SetString(PyExc_ValueError, "MKLNdarray not initialized");
        return NULL;
    }
    PyObject * rval = PyTuple_New(self->nd);
    if (rval == NULL)
    {
        return NULL;
    }
    for (int i = 0; i < self->nd; i++)
    {
        if (PyTuple_SetItem(rval, i, PyInt_FromLong(MKLNdarray_USER_DIMS(self)[i])))
        {
            Py_XDECREF(rval);
            return NULL;
        }
    }
    // printf("exit MKLNdarray_get_shape: refcnt = %ld \n", Py_REFCNT(self));
    return rval;
}


static PyObject*
MKLNdarray_get_dtype(MKLNdarray * self, void * closure)
{
    if (self->nd < 0 || self->user_structure == NULL || self->dtype < 0)
    {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray not initialized");
        return NULL;
    }

    PyObject * rval = PyString_FromFormat("dtype: %s", MKL_TYPE[self->dtype]);
    return rval;
}


static PyObject*
MKLNdarray_get_ndim(MKLNdarray * self, void * closure)
{
    return PyInt_FromLong(self->nd);
}


static PyObject*
MKLNdarray_get_size(MKLNdarray * self, void * closure)
{
    size_t total_element = 1;
    if (self->nd <= 0)
    {
        total_element = 0;
    }
    else
    {
        for (int i = 0; i < self->nd; i++)
        {
            total_element *= self->user_structure[i];
        }
    }
    return PyInt_FromLong(total_element);
}


static PyObject*
MKLNdarray_get_base(MKLNdarray * self, void * closure)
{
    PyObject * base = self->base;
    if (!base)
    {
        base = Py_None;
    }
    Py_INCREF(base);
    return base;
}


PyObject* MKLNdarray_CreateArrayObj(MKLNdarray* self)
{
    if (self->nd < 0 ||
        self->mkl_data == NULL ||
        self->mkl_layout == NULL)
    {
        PyErr_SetString(PyExc_ValueError, "Can't convert from a uninitialized MKLNdarray");
        return NULL;
    }

    npy_intp * npydims = (npy_intp*)malloc(self->nd * sizeof(npy_intp));
    assert (npydims);
    for (int i = 0; i < self->nd; i++)
    {
        npydims[i] = (npy_intp)self->user_structure[i];
    }

    PyArrayObject* rval = NULL;
    if (self->dtype == 12)
    {
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT64);
    }
    else
    {
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT32);
    }

    if (!rval)
    {
        free (npydims);
        npydims = NULL;
        return NULL;
    }
    free(npydims);
    npydims = NULL;

    void* rval_data = PyArray_DATA(rval);
    dnnLayout_t layout_user = NULL;
    int status = -1;
    dnnPrimitive_t primitive = NULL;
    if (self->dtype == 12)  // float64
    {
        status = dnnLayoutCreate_F64(&layout_user,
                                     self->nd,
                                     self->user_structure,
                                     self->user_structure + self->nd);

        if (status != 0 || layout_user == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionCreate_F64(&primitive, self->mkl_layout, layout_user);
        if (status != 0 || primitive == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnConversionCreate_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionExecute_F64(primitive, (void*)self->mkl_data, (void*)rval_data);
        if (status != 0)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnExecute_F64 failed");
            Py_DECREF(rval);
            return NULL;
        }
    }
    else  // float32
    {
        status = dnnLayoutCreate_F32(&layout_user,
                                     self->nd,
                                     self->user_structure,
                                     self->user_structure + self->nd);

        if (status != 0 || layout_user == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnLayoutCreate_F32 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionCreate_F32(&primitive, self->mkl_layout, layout_user);

        if (status != 0 || primitive == NULL)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnConversionCreate_F32 failed");
            Py_DECREF(rval);
            return NULL;
        }

        status = dnnConversionExecute_F32(primitive, (void*)self->mkl_data, (void*)rval_data);
        if (status != 0)
        {
            PyErr_Format(PyExc_RuntimeError, "MKLNdarray_CreateArrayObj: dnnExecute_F32 failed");
            Py_DECREF(rval);
            return NULL;
        }
    }

    return (PyObject*)rval;
}


PyObject* MKLNdarray_Zeros(PyObject* _unused, PyObject* args)
{
    if (!args)
    {
        PyErr_SetString(PyExc_TypeError, "MKLNdarray_Zeros: function takes at least 1 argument");
        return NULL;
    }

    PyObject* shape = NULL;
    int typenum = -1;

    if (!PyArg_ParseTuple(args, "O|i", &shape, &typenum))
    {
        printf("MKLNdarray_Zeros: PyArg_ParseTuple failed \n");
        return NULL;
    }

    if (typenum != 11 && typenum != 12)
    {
        printf("No dtype is specified. Use float32 as default. \n");
        typenum = 11;
    }

    if (!PySequence_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError, "shape argument must be a sequence");
        return NULL;
    }

    int shplen = PySequence_Length(shape);
    if (shplen <= 0)
    {
        PyErr_SetString(PyExc_TypeError, "length of shape argument must >= 1");
        return NULL;
    }

    size_t* newdims = (size_t*)malloc(sizeof(size_t) * shplen);
    if (!newdims)
    {
        PyErr_SetString(PyExc_MemoryError, "MKLNdarray_Zeros: failed to allocate temp space");
        return NULL;
    }

    for (int i = shplen -1; i >= 0; i--)
    {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if (shp_el_obj == NULL)
        {
            PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Zeros: index out of bound in sequence");
            free (newdims);
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        if (shp_el < 0)
        {
            PyErr_SetString(PyExc_ValueError,
                    "MKLNdarray_Zeros: shape must contain only non-negative values for size of a dimension");
            free (newdims);
            return NULL;
        }
        newdims[i] = (size_t)shp_el;
    }

    PyObject* rval = MKLNdarray_ZEROS(shplen, newdims, typenum);
    free (newdims);
    return (PyObject*)rval;
}


/* type:tp_methods
 * Describe methos of a type. ml_name/ml_meth/ml_flags/ml_doc.
 * ml_name: name of method
 * ml_meth: PyCFunction, point to the C implementation
 * ml_flags: indicate how the call should be constructed
 * ml_doc: docstring
 */
static PyMethodDef MKLNdarray_methods[] = {

    {"__array__",
        (PyCFunction)MKLNdarray_CreateArrayObj, METH_VARARGS,
        "Copy from MKL to a numpy ndarray."},
    /*
    {"__copy__",
        (PyCFunction)MKLNdarray_View, METH_NOARGS,
        "Create a shallow copy of this object. Used by module copy"},
    */
    {"zeros",
        (PyCFunction)MKLNdarray_Zeros, METH_STATIC | METH_VARARGS,
        "Create a new MklNdarray with specified shape, filled with zeros."},
    /*
    {"copy",
        (PyCFunction)MKLNdarray_Copy, METH_NOARGS,
        "Create a copy of this object."},
    */

    {NULL, NULL, 0, NULL}  /* Sentinel */
};


/* type:tp_members
 * Describe attributes of a type. name/type/offset/flags/doc.
 */
static PyMemberDef MKLNdarray_members[] = {
    {NULL}      /* Sentinel */
};


/* type:tp_getset
 * get/set attribute of instances of this type. name/getter/setter/doc/closure
 */
static PyGetSetDef MKLNdarray_getset[] = {
    {"shape",
        (getter)MKLNdarray_get_shape,
        NULL,
        "shape of this ndarray (tuple)",
        NULL},

    {"dtype",
        (getter)MKLNdarray_get_dtype,
        NULL,
        "the dtype of the element.",
        NULL},

    {"size",
        (getter)MKLNdarray_get_size,
        NULL,
        "the number of elements in this object.",
        NULL},

    {"ndim",
        (getter)MKLNdarray_get_ndim,
        NULL,
        "the number of dimensions in this objec.",
        NULL},

    {"base",
        (getter)MKLNdarray_get_base,
        NULL,
        "if this ndarray is a view, base is the original ndarray.",
        NULL},

    {NULL, NULL, NULL, NULL}  /* Sentinel*/
};

/* type object.
 * If you want to define a new object type, you need to create a new type object.
 */
static PyTypeObject MKLNdarrayType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "MKLNdarray",              /*tp_name*/
    sizeof(MKLNdarray),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MKLNdarray_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    MKLNdarray_repr,           /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
#endif
    "MKLNdarray objects",      /*tp_doc */
    0,                         /*tp_traverse*/
    0,                         /*tp_clear*/
    0,                         /*tp_richcompare*/
    0,                         /*tp_weaklistoffset*/
    0,                         /*tp_iter*/
    0,                         /*tp_iternext*/
    MKLNdarray_methods,        /*tp_methods*/
    MKLNdarray_members,        /*tp_members*/
    MKLNdarray_getset,         /*tp_getset*/
    0,                         /*tp_base*/
    0,                         /*tp_dict*/
    0,                         /*tp_descr_get*/
    0,                         /*tp_descr_set*/
    0,                         /*tp_dictoffset*/
    (initproc)MKLNdarray_init, /*tp_init*/
    0,                         /*tp_alloc*/
    MKLNdarray_new,            /*tp_new*/
};


int MKLNdarray_Check(const PyObject* ob)
{
    return ((Py_TYPE(ob) == &MKLNdarrayType) ? 1 : 0);
}


/*
 * [Re]allocate a MKLNdarray with access to 'nd' dimensions.
 *
 * Note: This does not allocate storage for data, or free pre-exsiting storage.
 */
int MKLNdarray_set_nd(MKLNdarray* self, const int nd)
{
    if (nd != self->nd)
    {
        if (self->user_structure)
        {
            free(self->user_structure);
            self->user_structure = NULL;
        }

        self->user_structure = (size_t*)malloc(2 * nd * sizeof (size_t));
        if (NULL == self->user_structure)
        {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate dim or str");
            return -1;
        }
        //initialize all dimensions and strides to 0
        for (int i = 0; i < 2 * nd; i++)
        {
            self->user_structure[i] = 0;
        }
        self->nd = nd;
    }
    return 0;
}


PyObject*
MKLNdarray_New(int nd, int typenum)
{
    MKLNdarray* self = (MKLNdarray*)(MKLNdarrayType.tp_alloc(&MKLNdarrayType, 0));
    if (self == NULL)
    {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_New failed to allocate self");
        return NULL;
    }

    self->base = NULL;
    self->dtype = typenum;
    self->user_structure = NULL;
    self->mkl_data = NULL;
    self->mkldata_size = 0;
    self->mkl_layout = NULL;

    if (nd == 0)
    {
        self->nd = nd;
    }
    else if (nd > 0)
    {
        if (0 != MKLNdarray_set_nd(self, nd))
        {
            Py_DECREF(self);
            return NULL;
        }
    }
    else
    {
        self->nd = -1;
    }
    return (PyObject*)self;
}



/* Declare methods belong to this module but not in MKLNdarray.
 * Used in module initialization function.
 */
static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}   /* Sentinel */
};

/*
 * Module initialization function.
 * TODO: Just for Python2.X. Need to add support for Python3.
 */
PyMODINIT_FUNC
initmkl_ndarray(void)
{
    import_array();
    PyObject* m = NULL;

    if (PyType_Ready(&MKLNdarrayType) < 0)
    {
        printf("MKLNdarrayType failed \n");
        return;
    }

    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float32", PyInt_FromLong(11));
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float64", PyInt_FromLong(12));

    m = Py_InitModule3("mkl_ndarray", module_methods, "MKL implementation of a ndarray object.");
    if (m == NULL)
    {
        printf("Py_InitModule3 failed to init mkl_ndarray. \n");
        return;
    }
    Py_INCREF(&MKLNdarrayType);
    PyModule_AddObject(m, "MKLNdarray", (PyObject*)&MKLNdarrayType);

    return;
}
