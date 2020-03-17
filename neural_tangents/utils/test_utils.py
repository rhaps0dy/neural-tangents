# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for testing."""

import logging
from jax.api import jit
from jax.api import vmap
from jax.lib import xla_bridge
import jax.numpy as np
import jax.test_util as jtu


def _jit_vmap(f):
  return jit(vmap(f))


def update_test_tolerance(f32_tol=5e-3, f64_tol=1e-5):
  jtu._default_tolerance[np.onp.dtype(np.onp.float32)] = f32_tol
  jtu._default_tolerance[np.onp.dtype(np.onp.float64)] = f64_tol
  def default_tolerance():
    if jtu.device_under_test() != 'tpu':
      return jtu._default_tolerance
    tol = jtu._default_tolerance.copy()
    tol[np.onp.dtype(np.onp.float32)] = 5e-2
    return tol
  jtu.default_tolerance = default_tolerance


def stub_out_pmap(batch, count):
  # If we are using GPU or CPU stub out pmap with vmap to simulate multi-core.
  if count > 0:

    class xla_bridge_stub(object):

      def device_count(self):
        return count

    platform = xla_bridge.get_backend().platform
    if platform == 'gpu' or platform == 'cpu':
      batch.pmap = _jit_vmap
      batch.xla_bridge = xla_bridge_stub()


def _log(relative_error, expected, actual, did_pass):
  msg = 'PASSED' if did_pass else 'FAILED'
  logging.info(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
               f'\n{msg} with {relative_error} relative error: \n'
               f'---------------------------------------------\n'
               f'EXPECTED: \n'
               f'{expected}\n'
               f'---------------------------------------------\n'
               f'ACTUAL: \n'
               f'{actual}\n'
               f'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n'
               )


def assert_close_matrices(self, expected, actual, rtol):
  self.assertEqual(expected.shape, actual.shape)
  relative_error = (
      np.linalg.norm(actual - expected) /
      np.maximum(np.linalg.norm(expected), 1e-12))
  if relative_error > rtol or np.isnan(relative_error):
    _log(relative_error, expected, actual, False)
    self.fail(self.failureException('Relative ERROR: ',
                                    float(relative_error),
                                    'EXPECTED:' + ' ' * 50,
                                    expected,
                                    'ACTUAL:' + ' ' * 50,
                                    actual))
  else:
    _log(relative_error, expected, actual, True)
