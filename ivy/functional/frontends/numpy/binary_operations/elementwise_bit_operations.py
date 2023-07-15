import ivy
import ivy.functional.frontends.numpy as np_frontend

from ivy.functional.frontends.numpy import promote_types_of_numpy_inputs

from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
    handle_numpy_dtype,
    from_zero_dim_arrays_to_scalar,
    handle_numpy_out,
)

@handle_numpy_out
@handle_numpy_dtype
@to_ivy_arrays_and_back
@handle_numpy_casting
@from_zero_dim_arrays_to_scalar
def left_shift(x1,
               x2,
               /,
               out=None,
               *,
               where=True,
               casting='same_kind',
               order='K',
               dtype=None):
    x1, x2 = promote_types_of_numpy_inputs(x1, x2)
    ret = ivy.bitwise_left_shift(x1, x2, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret
