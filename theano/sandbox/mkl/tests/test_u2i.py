


import theano
from theano import tensor
from theano.sandbox.mkl import mkl_type
from theano.sandbox.mkl.mkl_type import MKLNdarrayType

import u2i_op
import mkl_ndarray
import numpy


def test_i2u():

    x = MKLNdarrayType(broadcastable=(True, True), dtype='float32')('x')
    y = u2i_op.I2U_Op()(x)

    f = theano.function(inputs=[x], outputs=y)
    # theano.printing.pydotprint(f, outfile='1.png', var_with_name_simple=True)

    a = mkl_ndarray.mkl_ndarray.MKLNdarray.zeros((2,2), mkl_ndarray.mkl_ndarray.MKLNdarray.float32)
    # print(a)
    o = f(a)

    assert isinstance(o, numpy.ndarray)
    assert str(o.dtype) == 'float32'


def test_u2i():
    x = tensor.fmatrix('x')
    y = u2i_op.U2I_BN(eps=0.5)(x)

    f = theano.function(inputs=[x], outputs=y)
    # theano.printing.pydotprint(f, outfile='2.png', var_with_name_simple=True)

    a = numpy.random.rand(2, 2).astype(numpy.float32)
    o = f(a)

    assert isinstance(o, mkl_ndarray.mkl_ndarray.MKLNdarray)
    assert numpy.allclose(a, o.__array__())
    print(o.dtype)


if __name__ == "__main__":
    test_i2u()
    test_u2i()

