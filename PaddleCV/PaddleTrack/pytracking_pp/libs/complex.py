import numpy as np
from pytracking_pp.libs.tensorlist import tensor_operation


def is_complex(a: np.array) -> bool:
    return a.ndim >= 4 and a.shape[-1] == 2


def is_real(a: np.array) -> bool:
    return not is_complex(a)


@tensor_operation
def mult(a: np.array, b: np.array):
    """Pointwise complex multiplication of complex tensors."""

    if is_real(a):
        if a.ndim >= b.ndim:
            raise ValueError('Incorrect dimensions.')
        # a is real
        return mult_real_cplx(a, b)
    if is_real(b):
        if b.ndim >= a.ndim:
            raise ValueError('Incorrect dimensions.')
        # b is real
        return mult_real_cplx(b, a)

    # Both complex
    c = mult_real_cplx(a[..., 0], b)
    c[..., 0] -= a[..., 1] * b[..., 1]
    c[..., 1] += a[..., 1] * b[..., 0]
    return c


@tensor_operation
def mult_conj(a: np.array, b: np.array):
    """Pointwise complex multiplication of complex tensors, with conjugate on b: a*conj(b)."""

    if is_real(a):
        if a.ndim >= b.ndim:
            raise ValueError('Incorrect dimensions.')
        # a is real
        return mult_real_cplx(a, conj(b))
    if is_real(b):
        if b.ndim >= a.ndim:
            raise ValueError('Incorrect dimensions.')
        # b is real
        return mult_real_cplx(b, a)

    # Both complex
    c = mult_real_cplx(b[...,0], a)
    c[..., 0] += a[..., 1] * b[..., 1]
    c[..., 1] -= a[..., 0] * b[..., 1]
    return c


@tensor_operation
def mult_real_cplx(a: np.array, b: np.array):
    """Pointwise complex multiplication of real tensor a with complex tensor b."""

    if is_real(b):
        raise ValueError('Last dimension must have length 2.')

    return np.expand_dims(a, -1) * b


@tensor_operation
def div(a: np.array, b: np.array):
    """Pointwise complex division of complex tensors."""

    if is_real(b):
        if b.ndim >= a.ndim:
            raise ValueError('Incorrect dimensions.')
        # b is real
        return div_cplx_real(a, b)

    return div_cplx_real(mult_conj(a, b), abs_sqr(b))


@tensor_operation
def div_cplx_real(a: np.array, b: np.array):
    """Pointwise complex division of complex tensor a with real tensor b."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a / np.expand_dims(b, -1)


@tensor_operation
def abs_sqr(a: np.array):
    """Squared absolute value."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return np.sum(a*a, -1)


@tensor_operation
def abs(a: np.array):
    """Absolute value."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return np.sqrt(abs_sqr(a))


@tensor_operation
def conj(a: np.array):
    """Complex conjugate."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    # return a * np.array([1, -1], device=a.device)
    return complex(a[...,0], -a[...,1])


@tensor_operation
def real(a: np.array):
    """Real part."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a[..., 0]


@tensor_operation
def imag(a: np.array):
    """Imaginary part."""

    if is_real(a):
        raise ValueError('Last dimension must have length 2.')

    return a[..., 1]


@tensor_operation
def complex(a: np.array, b: np.array = None):
    """Create complex tensor from real and imaginary part."""

    if b is None:
        b = np.zeros(a.shape, a.dtype)
    elif a is None:
        a = np.zeros(b.shape, b.dtype)

    return np.concatenate((np.expand_dims(a, -1), np.expand_dims(b, -1)), -1)


@tensor_operation
def mtimes(a: np.array, b: np.array, conj_a=False, conj_b=False):
    """Complex matrix multiplication of complex tensors.
    The dimensions (-3, -2) are matrix multiplied. -1 is the complex dimension."""

    if is_real(a):
        if a.ndim >= b.ndim:
            raise ValueError('Incorrect dimensions.')
        return mtimes_real_complex(a, b, conj_b=conj_b)
    if is_real(b):
        if b.ndim >= a.ndim:
            raise ValueError('Incorrect dimensions.')
        return mtimes_complex_real(a, b, conj_a=conj_a)

    if not conj_a and not conj_b:
        return complex(np.matmul(a[..., 0], b[..., 0]) - np.matmul(a[..., 1], b[..., 1]),
                       np.matmul(a[..., 0], b[..., 1]) + np.matmul(a[..., 1], b[..., 0]))
    if conj_a and not conj_b:
        return complex(np.matmul(a[..., 0], b[..., 0]) + np.matmul(a[..., 1], b[..., 1]),
                       np.matmul(a[..., 0], b[..., 1]) - np.matmul(a[..., 1], b[..., 0]))
    if not conj_a and conj_b:
        return complex(np.matmul(a[..., 0], b[..., 0]) + np.matmul(a[..., 1], b[..., 1]),
                       np.matmul(a[..., 1], b[..., 0]) - np.matmul(a[..., 0], b[..., 1]))
    if conj_a and conj_b:
        return complex(np.matmul(a[..., 0], b[..., 0]) - np.matmul(a[..., 1], b[..., 1]),
                       -np.matmul(a[..., 0], b[..., 1]) - np.matmul(a[..., 1], b[..., 0]))


@tensor_operation
def mtimes_real_complex(a: np.array, b: np.array, conj_b=False):
    if is_real(b):
        raise ValueError('Incorrect dimensions.')

    if not conj_b:
        return complex(np.matmul(a, b[..., 0]), np.matmul(a, b[..., 1]))
    if conj_b:
        return complex(np.matmul(a, b[..., 0]), -np.matmul(a, b[..., 1]))


@tensor_operation
def mtimes_complex_real(a: np.array, b: np.array, conj_a=False):
    if is_real(a):
        raise ValueError('Incorrect dimensions.')

    if not conj_a:
        return complex(np.matmul(a[..., 0], b), np.matmul(a[..., 1], b))
    if conj_a:
        return complex(np.matmul(a[..., 0], b), -np.matmul(a[..., 1], b))


@tensor_operation
def exp_imag(a: np.array):
    """Complex exponential with imaginary input: e^(i*a)"""

    a = np.expand_dims(a, -1)
    return np.concatenate((np.cos(a), np.sin(a)), -1)



