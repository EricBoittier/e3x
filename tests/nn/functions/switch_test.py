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

import math

import e3x
import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize('x0, x1', [(0.0, 1.0), (-1.0, 2.0), (0.4, 0.6)])
def test_smooth_switch(x0: float, x1: float) -> None:
  x = jnp.linspace(x0, x1, num=11)
  expected_value = jnp.asarray([  # Same for both test cases.
      0.0000000e00,
      4.5352624e-04,
      3.7413392e-02,
      1.6116680e-01,
      3.2702142e-01,
      5.0000000e-01,
      6.7297864e-01,
      8.3883321e-01,
      9.6258658e-01,
      9.9954647e-01,
      1.0000000e00,
  ])
  expected_grad = jnp.asarray([
      0.0000000,
      0.0397434,
      0.8284502,
      1.5398244,
      1.7206353,
      1.7320508,
      1.7206357,
      1.5398246,
      0.8284502,
      0.0397433,
      0.0000000,
  ]) / (x1 - x0)
  value, grad = e3x.ops.evaluate_derivatives(
      lambda x: e3x.nn.smooth_switch(x, x0=x0, x1=x1), x, max_order=1
  )
  assert jnp.allclose(value, expected_value, atol=1e-5)
  assert jnp.allclose(grad, expected_grad, atol=1e-5)


def test_smooth_switch_has_nan_safe_derivatives() -> None:
  finfo = jnp.finfo(jnp.float32)
  x = jnp.asarray(
      [0.0, finfo.tiny, finfo.eps, 1 - finfo.epsneg, 1.0],
  )
  for y in e3x.ops.evaluate_derivatives(e3x.nn.smooth_switch, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))


def test_smooth_switch_has_finite_derivatives_everywhere() -> None:
  # In float32, exp(√3/2 (1/x - 1/(1-x))) overflows for x < 0.0102: values and
  # derivatives of any order (forward and reverse mode) must stay finite.
  x = jnp.concatenate([
      jnp.linspace(-0.1, 1.1, num=1201, dtype=jnp.float32),
      jnp.geomspace(1e-7, 0.02, num=200, dtype=jnp.float32),
      1 - jnp.geomspace(1e-7, 0.02, num=200, dtype=jnp.float32),
  ])
  for y in e3x.ops.evaluate_derivatives(e3x.nn.smooth_switch, x, max_order=4):
    assert jnp.all(jnp.isfinite(y))
  # Reverse-over-reverse and Hessians through other operations used to give
  # NaN (inf * 0), e.g. for a distance of 1.05 in a 1-10 switch.
  f = lambda r: jnp.sum(e3x.nn.smooth_switch(r, x0=1.0, x1=10.0) / r)
  r = 1.0 + 9.0 * x[::8]
  assert jnp.all(jnp.isfinite(jax.grad(f)(r)))
  assert jnp.all(jnp.isfinite(jax.hessian(f)(r)))
  assert jnp.all(jnp.isfinite(jax.jacrev(jax.grad(f))(r)))

  positions = jnp.asarray([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0], [0.0, 1.5, 0.2]])
  distances = lambda p: jnp.linalg.norm(
      p[:, None] - p[None] + jnp.eye(3)[..., None], axis=-1  # no zeros
  )
  energy = lambda p: f(distances(p))
  assert jnp.all(jnp.isfinite(jax.hessian(energy)(positions)))


def test_smooth_switch_values_far_from_midpoint() -> None:
  x = np.geomspace(1e-3, 0.5, num=500, dtype=np.float32)
  x = np.concatenate([x, 1 - x])
  # Reference in float64: 1 / (1 + exp(a)) and its derivative, without 1 - s.
  x64 = x.astype(np.float64)
  a = math.sqrt(3) / 2 * (1 / x64 - 1 / (1 - x64))
  with np.errstate(over='ignore'):
    s, t = 1 / (1 + np.exp(a)), 1 / (1 + np.exp(-a))
  ds = s * t * math.sqrt(3) / 2 * (1 / x64**2 + 1 / (1 - x64) ** 2)
  value, grad = e3x.ops.evaluate_derivatives(
      e3x.nn.smooth_switch, jnp.asarray(x), max_order=1
  )
  assert np.allclose(value, s, rtol=1e-5, atol=1e-30)
  assert np.allclose(grad, ds, rtol=1e-4, atol=1e-30)


@pytest.mark.parametrize('x0, x1', [(1.0, 0.0), (0.5, 0.5)])
def test_smooth_switch_raises_with_invalid_x0_x1(x0: float, x1: float) -> None:
  with pytest.raises(ValueError, match='x1 must be larger than x0'):
    e3x.nn.smooth_switch(1, x0=x0, x1=x1)
