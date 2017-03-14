
import theano
from theano import gof, tensor
from theano.sandbox.mkl import mkl_type
from theano.sandbox.mkl.mkl_type import MKLNdarrayType

class I2U_Op(gof.Op):
    __props__ = ()

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return "I2U"

    def make_node(self, x):
        if not isinstance(x.type, MKLNdarrayType):
            raise TypeError('Expected a Theano variable with type MKLNdarrayType.')
        return gof.Apply(self, [x], [tensor.TensorType(broadcastable=x.broadcastable, dtype=x.dtype)()])

    def c_code(self, node, name, inputs, outputs, sub):
        inp = inputs[0]
        out = outputs[0]
        fail = sub['fail']
        return """
        Py_XDECREF(%(out)s);
        %(out)s = (PyArrayObject*)MKLNdarray_CreateArrayObj(%(inp)s);
        if (!%(out)s)
        {
            %(fail)s;
        }
        """ % locals()

    def c_code_cache_version(self):
        return (1, 0, 0)


from theano.sandbox.mkl.basic_ops import BaseConvertOp
class U2I_BN(BaseConvertOp):
    __props__ = ('eps',)

    def __init__(self, eps=1e-5):
        self.eps = eps

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def c_support_code(self):
        ccode = """
            #define DIMENSION 4
            #define CHECK_ERR(f, err) \\
                    do { \\
                        (err) = (f); \\
                        if ((err) != E_SUCCESS) { \\
                            printf("Error in file [%s:%d], err code (%d)", \\
                                    __FILE__, __LINE__, err); \\
                            exit(1); \\
                        } \\
                    } while(0)
        """
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        eps = self.eps
        fail = sub['fail']

        if 'float32' == node.inputs[0].type.dtype:
            typenum = 11
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            typenum = 12
            precision = 'F64'
        else:
            raise Exception('Type %s is not supported!' % node.inputs[0].type.dtype)

        ccode = """
        int ndim = PyArray_NDIM(%(x)s);
        int dtype = PyArray_TYPE(%(x)s);
        npy_intp* d = PyArray_DIMS(%(x)s);

        assert (dtype == %(typenum)s);

        size_t* dims = (size_t*)malloc(ndim * sizeof (size_t));

        for (int i = 0; i < ndim; i++)
        {
            dims[i] = (size_t)d[i];
        }

        Py_XDECREF(%(z)s);
        %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
        
        if (!%(z)s)
        {
            %(fail)s;
        }

        int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
        if (status != 0)
        {
            free (dims);
            %(fail)s;
        }

        free (dims);
        CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, ndim, %(z)s->user_structure, %(z)s->user_structure + ndim), err);
        CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&primitive, NULL, layout_user, %(eps)s), err);
        CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&(%(z)s->mkl_layout), primitive, dnnResourceSrc), err);

        if (!dnnLayoutCompare_%(precision)s(layout_user, layout_internal)) {
            if (NULL == to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, %(z)s->mkl_layout), err);
            }
        }

        CHECK_ERR( dnnAllocateBuffer_%(precision)s(&(%(z)s->mkl_data), %(z)s->mkl_layout), err);

        if (to_internal)
        {
            CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, PyArray_DATA(%(x)s), %(z)s->mkl_data), err);
        }
        else
        {
            memcpy(%(z)s->mkl_data, PyArray_DATA(%(x)s), dnnLayoutGetMemorySize_%(precision)s(%(z)s->mkl_layout));
        }

        // printf("BN CONVERT OK \\n");
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (1, 0, 0)
