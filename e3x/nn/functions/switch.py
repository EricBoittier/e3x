# Copyright 2026 The e3x Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Switch functions.

Functions that are :math:`0` for :math:`x<a`, :math:`1` for :math:`x>b`, and
interpolate between :math:`0` and :math:`1` on the interval :math:`[a,b]`.
"""

import functools
import math
from typing import Any, Tuple, Union
import jax
import jax.numpy as jnp
import jaxtyping

Array = jaxtyping.Array
Float = jaxtyping.Float

# Most gradual slope possible without slope artifacts.
_SMOOTH_SWITCH_SLOPE = math.sqrt(3) / 2


def _smooth_switch_safe(
    x: Float[Array, '...'],
) -> Tuple[Float[Array, '...'], Float[Array, '...']]:
  """Helper function for smooth_switch and its JVP implementation.

  Returns the switch value and its derivative. The exponential is written as
  logistic functions, 1/(1 + exp(a)) = sigmoid(-a) and its derivative with
  sigmoid(a) * sigmoid(-a), so nothing overflows: exp(a) is infinite in float32
  for x < 0.0102. Where both are constant in floating point (0 or 1, and 0),
  they are computed at x = 0.5 and masked, so that derivatives of any order of
  the unused branch stay finite and can not turn into NaN through `jnp.where`.

  Args:
    x: Input array, already rescaled to the interval [0, 1].

  Returns:
    The tuple (value, derivative).
  """
  # Mask numerically problematic values.
  eps = jnp.finfo(x.dtype).epsneg
  small = x < eps
  large = x > 1 - eps
  outside = jnp.logical_or(small, large)
  safe_x = jnp.where(outside, 0.5, x)
  a = _SMOOTH_SWITCH_SLOPE * (1 / safe_x - 1 / (1 - safe_x))
  # Far from the midpoint, the derivative underflows to zero while 1/x**2 can
  # get large: mask these points too.
  mask = jnp.logical_or(
      outside, jax.nn.sigmoid(a) * jax.nn.sigmoid(-a) == 0
  )
  safe_x = jnp.where(mask, 0.5, x)
  a = _SMOOTH_SWITCH_SLOPE * (1 / safe_x - 1 / (1 - safe_x))
  value = jnp.where(
      mask, jnp.where(x < 0.5, 0, 1).astype(x.dtype), jax.nn.sigmoid(-a)
  )
  derivative = jnp.where(
      mask,
      0,
      _SMOOTH_SWITCH_SLOPE
      * jax.nn.sigmoid(a)
      * jax.nn.sigmoid(-a)
      * (1 / safe_x**2 + 1 / (1 - safe_x) ** 2),
  )
  return value, derivative


@functools.partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def smooth_switch(
    x: Float[Array, '...'],
    x0: Union[Float[Array, ''], float] = 0.0,
    x1: Union[Float[Array, ''], float] = 1.0,
) -> Float[Array, '...']:
  r"""Smooth switch function.

  Computes the function

  .. math::
    \mathrm{smooth\_switch}(x) = \begin{cases}
      0, & x \le x_0\\
      \left(1+\exp\left(\frac{\sqrt{3}}{2} \left( \frac{x_1-x_0}{x-x_0} -
      \frac{x_1-x_0}{x-x_1}\right)\right)\right)^{-1}, & x_0 < x < x_1\\
      1, & x \ge x_1
    \end{cases}

  which interpolates between :math:`0` and :math:`1` in the interval
  :math:`[x_0, x_1]` and is infinitely differentiable.

  .. jupyter-execute::
    :hide-code:

    import numpy as np, matplotlib.pyplot as plt
    import matplotlib_inline.backend_inline as inl
    from e3x.nn import smooth_switch
    inl.set_matplotlib_formats('pdf', 'svg')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    x = np.linspace(-2, 2, num=1001)
    y1 = smooth_switch(x, x0=0.0, x1=1.0)
    y2 = smooth_switch(x, x0=-1.5, x1=0.5)
    plt.xlabel(r'$x$'); plt.ylabel(r'$\mathrm{smooth\_switch}(x)$')
    plt.plot(x, y1, lw=3, ls='-',  label='x0=0.0, x1=1.0')
    plt.plot(x, y2, lw=3, ls='--', label='x0=-1.5, x1=0.5')
    plt.legend(); plt.grid()

  Args:
    x: Input array.
    x0: Values below are zero.
    x1: Values above are one.

  Returns:
    The function value.
  """
  # Check for valid boundaries.
  if x0 >= x1:
    raise ValueError(
        f'input x1 must be larger than x0, received x0={x0} and x1={x1}'
    )

  # Rescale x if necessary.
  if x0 != 0.0 or x1 != 1.0:
    x = (x - x0) / (x1 - x0)

  value, _ = _smooth_switch_safe(jnp.asarray(x))
  return value


@smooth_switch.defjvp
def _smooth_switch_jvp_impl(
    x0: Union[Float[Array, ''], float],
    x1: Union[Float[Array, ''], float],
    primals: Any,
    tangents: Any,
) -> Tuple[Any, Any]:
  """JVP implementation for smooth_switch."""
  (x,) = primals
  (x_dot,) = tangents

  # Rescale x and x_dot if necessary.
  if x0 != 0.0 or x1 != 1.0:
    x = (x - x0) / (x1 - x0)
    x_dot = x_dot / (x1 - x0)

  primal_out, derivative = _smooth_switch_safe(x)
  tangent_out = derivative * x_dot
  return primal_out, tangent_out
