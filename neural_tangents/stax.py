# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License');
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

"""Analytic NNGP and NTK library.

This library contains layer constructors mimicking those in
`jax.experimental.stax` with similar API apart apart from:

1) Instead of `(init_fn, apply_fn)` tuple, layer constructors return a triple \
  `(init_fn, apply_fn, kernel_fn)`, where the added `kernel_fn` maps an \
  `Kernel` to a new `Kernel`, and represents the change in the \
  analytic NTK and NNGP kernels (`Kernel.nngp`, `Kernel.ntk`). These functions \
  are chained / stacked together within the `serial` or `parallel` \
  combinators, similarly to `init_fn` and `apply_fn`.

2) In layers with random weights, NTK parameterization is used by default \
  (https://arxiv.org/abs/1806.07572, page 3). Standard parameterization \
  (https://arxiv.org/abs/2001.07301) can be specified for `Conv` and `Dense` \
  layers by a keyword argument `parameterization`.

3) Some functionality may be missing (e.g. `BatchNorm`), and some may be \
  present only in our library (e.g. `CIRCULAR` padding, `LayerNorm`, \
  `GlobalAvgPool`, `GlobalSelfAttention` etc.).

Example:
  >>> from jax import random
  >>> import neural_tangents as nt
  >>> from neural_tangents import stax
  >>>
  >>> key1, key2 = random.split(random.PRNGKey(1), 2)
  >>> x_train = random.normal(key1, (20, 32, 32, 3))
  >>> y_train = random.uniform(key1, (20, 10))
  >>> x_test = random.normal(key2, (5, 32, 32, 3))
  >>>
  >>> init_fn, apply_fn, kernel_fn = stax.serial(
  >>>     stax.Conv(128, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(256, (3, 3)),
  >>>     stax.Relu(),
  >>>     stax.Conv(512, (3, 3)),
  >>>     stax.Flatten(),
  >>>     stax.Dense(10)
  >>> )
  >>>
  >>> # (5, 10) np.ndarray NNGP test prediction
  >>> y_test_nngp = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
  >>>                                       get='nngp')
  >>>
  >>> # (5, 10) np.ndarray NTK prediction
  >>> y_test_ntk = nt.predict.gp_inference(kernel_fn, x_train, y_train, x_test,
  >>>                                      get='ntk')
"""

import string
import functools
import operator as op
import warnings
import enum
from jax import lax
from jax import linear_util as lu
from jax import ops
from jax.abstract_arrays import ShapedArray
from jax.api_util import flatten_fun
import jax.experimental.stax as ostax
import jax.numpy as np
from jax.scipy.special import erf
from neural_tangents.utils.kernel import Kernel
import frozendict
from jax import random
import jax.interpreters.partial_eval as pe
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.nn import initializers
from neural_tangents.utils import utils


class Padding(enum.Enum):
  CIRCULAR = 'CIRCULAR'
  SAME = 'SAME'
  VALID = 'VALID'


class Pooling(enum.Enum):
  AVG = 'AVG'
  SUM = 'SUM'


def _randn(stddev=1e-2):
  """`jax.experimental.stax.randn` for implicitly-typed results."""
  def init(rng, shape):
    return stddev * random.normal(rng, shape)
  return init


def layer(layer_fns):
  """A convenience decorator to be added to all public layers like `Relu` etc.

  Makes the `kernel_fn` of the layer work with both input `np.ndarray`s (when
  the layer is the first one applied to inputs), and with `Kernel` for
  intermediary layers. Also adds optional arguments to the `kernel_fn` to
  allow specifying the computation and returned results with more flexibility.

  Args:
    layer: A layer function returning a triple `(init_fn, apply_fn, kernel_fn)`.

  Returns:
    A function with the same signature as `layer` with `kernel_fn` now
    accepting `np.ndarray`s as inputs if needed, and accepts optional `get`,
    `diagonal_batch`, `diagonal_spatial` arguments.
  """
  name = layer_fns.__name__

  @utils.wraps(layer_fns)
  def new_layer_fns(*args, **kwargs):
    init_fn, apply_fn, kernel_fn = layer_fns(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn(init_fn, kernel_fn)
    init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
    return init_fn, apply_fn, kernel_fn
  return new_layer_fns


def requires(**static_reqs):
  """Returns a decorator that augments `kernel_fn` with consistency checks.

  Use this to specify your `kernel_fn` input kernel requirements.
  """

  def req(kernel_fn):
    """Returns `kernel_fn` with additional consistency checks."""

    @utils.wraps(kernel_fn)
    def new_kernel_fn(kernels, **user_reqs):
      """Executes `kernel_fn` on `kernels` after checking consistency."""

      fused_reqs = _fuse_reqs(static_reqs, {}, **user_reqs)

      # `FanInConcat / FanInSum` have no requirements and
      # execute custom consistency checks.
      if not isinstance(kernels, list):
        for k, v in fused_reqs.items():
          if v is not None:  # `None` is treated as explicitly not having a req.
            if k in ('diagonal_batch', 'diagonal_spatial'):
              if getattr(kernels, k) and not v:
                raise ValueError(f'{kernel_fn} requires `{k} == {v}`, but input'
                                 f' kernel has `{k} == True`, hence does not '
                                 f'contain sufficient information. Please '
                                 f'recompute the input kernel with '
                                 f'`{k} == {v}`.')
            elif k in ('batch_axis', 'channel_axis'):
              ndim = len(kernels.shape1)
              v_kernel = getattr(kernels, k)
              v_pos = v % ndim
              if v_kernel != v_pos:
                raise ValueError(f'{kernel_fn} requires `{k} == {v_pos}`, but '
                                 f'input kernel has `{k} == {v_kernel}`, making'
                                 f' the infinite limit ill-defined.')
            else:
              raise NotImplementedError(k)

      return kernel_fn(kernels)

    setattr(new_kernel_fn, _INPUT_REQ, frozendict.frozendict(static_reqs))
    return new_kernel_fn

  return req


# LAYERS


@layer
def serial(*layers):
  """Combinator for composing layers in serial.

  Based on `jax.experimental.stax.serial`.

  Args:
    :layers: a sequence of layers, each an `(init_fn, apply_fn, kernel_fn)`
      triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triple,
    representing the serial composition of the given sequence of layers.

  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn, apply_fn = ostax.serial(*zip(init_fns, apply_fns))

  @requires(**_get_input_req_attr(kernel_fns))
  def kernel_fn(kernels):
    for f in kernel_fns:
      kernels = f(kernels)
    return kernels

  return init_fn, apply_fn, kernel_fn


@layer
def parallel(*layers):
  """Combinator for composing layers in parallel.

  The layer resulting from this combinator is often used with the `FanOut` and
  `FanInSum`/`FanInConcat` layers. Based on `jax.experimental.stax.parallel`.

  Args:
    :layers: a sequence of layers, each an `(init_fn, apply_fn, kernel_fn)`
      triple.

  Returns:
    A new layer, meaning an `(init_fn, apply_fn, kernel_fn)` triples,
    representing the parallel composition of the given sequence of layers. In
    particular, the returned layer takes a sequence of inputs and returns a
    sequence of outputs with the same length as the argument `layers`.

  """
  init_fns, apply_fns, kernel_fns = zip(*layers)
  init_fn_stax, apply_fn = ostax.parallel(*zip(init_fns, apply_fns))

  def init_fn(rng, input_shape):
    return list(init_fn_stax(rng, input_shape))

  @requires(**_get_input_req_attr(kernel_fns))
  def kernel_fn(kernels):
    return [f(ker) for ker, f in zip(kernels, kernel_fns)]

  return init_fn, apply_fn, kernel_fn


@layer
def Dense(out_dim,
          W_std=1.,
          b_std=0.,
          W_init=_randn(1.0),
          b_init=_randn(1.0),
          parameterization='ntk',
          batch_axis=0,
          channel_axis=-1):
  r"""Layer constructor function for a dense (fully-connected) layer.

  Based on `jax.experimental.stax.Dense`. Has a similar API, apart from:

  `W_init` and `b_init` only change the behavior of the finite width network,
  and are not used by `kernel_fn`. In most cases, `W_std` and `b_std` should
  be used instead.

  Args:
    :parameterization: Either 'ntk' or 'standard'.

      Under ntk parameterization (https://arxiv.org/abs/1806.07572, page 3),
      weights and biases are initialized as :math:`W_{ij} \sim N(0,1)`,
      :math:`b_i \sim N(0,1)`, and the finite width layer equation is
      :math:`z_i = \sigma_W / sqrt(N) sum_j W_{ij} x_j + \sigma_b b_i`.

      Under standard parameterization (https://arxiv.org/abs/2001.07301),
      weights and biases are initialized as :math:`W_{ij} \sim N(0,W_std^2/N)`,
      :math:`b_i \sim N(0,\sigma_b^2)`, and the finite width layer equation is
      :math:`z_i = \sum_j W_ij x_j + b_i`.

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    `(init_fn, apply_fn, kernel_fn)`.

  """
  # TODO: after experimentation, evaluate whether to change default
  # parameterization from "ntk" to "standard"

  parameterization = parameterization.lower()

  def ntk_init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    output_shape = (input_shape[:_channel_axis] + (out_dim,)
                    + input_shape[_channel_axis + 1:])
    k1, k2 = random.split(rng)
    W = W_init(k1, (input_shape[_channel_axis], out_dim))
    b = b_init(k2, (out_dim,))
    return output_shape, (W, b)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    return output_shape, (W * W_std / np.sqrt(input_shape[channel_axis]),
                          b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, **kwargs):
    W, b = params

    prod = np.moveaxis(np.tensordot(W, inputs, (0, channel_axis)),
                       0, channel_axis)

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(inputs.shape[channel_axis])
      res = norm * prod + b_std * b
    elif parameterization == 'standard':
      res = prod  + b

    return res

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(kernels):
    """Compute the transformed kernels after a dense layer."""
    cov1, nngp, cov2, ntk = \
      kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

    def fc(x):
      return _affine(x, W_std, b_std)

    if parameterization == 'ntk':
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))
      if ntk is not None:
        ntk = nngp + W_std**2 * ntk
    elif parameterization == 'standard':
      input_width = kernels.shape1[channel_axis]
      if ntk is not None:
        ntk = input_width * nngp + 1. + W_std**2 * ntk
      cov1, nngp, cov2 = map(fc, (cov1, nngp, cov2))

    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk,
                            is_gaussian=True,
                            is_input=False)

  return init_fn, apply_fn, kernel_fn


@layer
def GeneralConv(dimension_numbers,
                out_chan,
                filter_shape,
                strides=None,
                padding=Padding.VALID.name,
                W_std=1.0,
                W_init=_randn(1.0),
                b_std=0.0,
                b_init=_randn(1.0),
                parameterization='ntk'):
  """Layer construction function for a general convolution layer.

  Based on `jax.experimental.stax.GeneralConv`. Has a similar API apart from:

  `W_init` and `b_init` only change the behavior of the finite width network,
    and are not used by `kernel_fn`. In most cases, `W_std` and `b_std` should
    be used instead.

  Args:
    :padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.
    :parameterization: Either "ntk" or "standard". These parameterizations are
      the direct analogues for convolution of the corresponding
      parameterizations for `Dense` layers.
  """

  parameterization = parameterization.lower()

  if dimension_numbers is None:
    spatial_dims = ''.join(c for c in string.ascii_uppercase
                           if c not in ('N' ,'C', 'I', 'O'))[:len(filter_shape)]
    lhs_spec = 'N' + spatial_dims + 'C'
    dimension_numbers = (lhs_spec, spatial_dims + 'IO', lhs_spec)

  lhs_spec = dimension_numbers[0]

  one = (1,) * len(filter_shape)
  strides = strides or one

  padding = Padding(padding)
  init_padding = padding
  if padding == Padding.CIRCULAR:
    init_padding = Padding.SAME

  def input_total_dim(input_shape):
    return input_shape[lhs_spec.index('C')] * np.prod(filter_shape)

  ntk_init_fn, _ = ostax.GeneralConv(dimension_numbers, out_chan, filter_shape,
                                     strides, init_padding.name, W_init, b_init)

  def standard_init_fn(rng, input_shape):
    output_shape, (W, b) = ntk_init_fn(rng, input_shape)
    norm = W_std / np.sqrt(input_total_dim(input_shape))
    return output_shape, (W * norm, b * b_std)

  if parameterization == 'ntk':
    init_fn = ntk_init_fn
  elif parameterization == 'standard':
    init_fn = standard_init_fn
  else:
    raise ValueError('Parameterization not supported: %s' % parameterization)

  def apply_fn(params, inputs, **kwargs):
    W, b = params

    if parameterization == 'ntk':
      norm = W_std / np.sqrt(input_total_dim(inputs.shape))
      b_rescale = b_std
    elif parameterization == 'standard':
      norm = 1.
      b_rescale = 1.

    apply_padding = padding
    if padding == Padding.CIRCULAR:
      apply_padding = Padding.VALID
      spatial_axes = tuple(dimension_numbers[0].index(c)
                           for c in dimension_numbers[1]
                           if c not in ('I', 'O'))
      inputs = _same_pad_for_filter_shape(inputs, filter_shape, strides,
                                          spatial_axes, 'wrap')

    return norm * lax.conv_general_dilated(
        inputs,
        W,
        strides,
        apply_padding.name,
        dimension_numbers=dimension_numbers) + b_rescale * b

  @requires(batch_axis=dimension_numbers[0].index('N'),
            channel_axis=dimension_numbers[0].index('C'))
  def kernel_fn(kernels):
    """Compute the transformed kernels after a conv layer."""
    cov1, nngp, cov2, ntk, is_reversed = (kernels.cov1, kernels.nngp,
                                          kernels.cov2, kernels.ntk,
                                          kernels.is_reversed)

    input_spec = tuple(c for c in dimension_numbers[0] if c not in ('N', 'C'))
    conv_spec = tuple(c for c in dimension_numbers[1] if c not in ('I', 'O'))
    input_to_filter_permutation = tuple(conv_spec.index(c) for c in input_spec)

    filter_shape_kernel = tuple(filter_shape[p] for p in
                                input_to_filter_permutation)
    strides_kernel = tuple(strides[p] for p in
                           input_to_filter_permutation)

    if kernels.diagonal_spatial:
      def conv_unscaled(x, batch_ndim):
        x = _conv_kernel_over_spatial(
            x, filter_shape_kernel, strides_kernel, padding, batch_ndim)
        return x

    else:
      if is_reversed:
        filter_shape_kernel = filter_shape_kernel[::-1]
        strides_kernel = strides_kernel[::-1]

      is_reversed = not is_reversed

      def conv_unscaled(x, batch_ndim):
        x = _conv_kernel(
            x, filter_shape_kernel, strides_kernel, padding, batch_ndim)
        return x

    def conv(x, batch_ndim):
      x = conv_unscaled(x, batch_ndim)
      return _affine(x, W_std, b_std)

    cov1 = conv(cov1, 1 if kernels.diagonal_batch else 2)
    cov2 = conv(cov2, 1 if kernels.diagonal_batch else 2)

    if parameterization == 'ntk':
      nngp = conv(nngp, 2)
      ntk = conv(ntk, 2) + nngp - b_std**2 if ntk is not None else ntk

    elif parameterization == 'standard':
      nngp_unscaled = conv_unscaled(nngp, 2)
      if ntk is not None:
        ntk = (
            input_total_dim(kernels.shape1) * nngp_unscaled + 1. +
            W_std**2 * conv_unscaled(ntk, 2))
      nngp = _affine(nngp_unscaled, W_std, b_std)

    res = kernels._replace(cov1=cov1,
                           nngp=nngp,
                           cov2=cov2,
                           ntk=ntk,
                           is_gaussian=True,
                           is_reversed=is_reversed,
                           batch_axis=dimension_numbers[2].index('N'),
                           channel_axis=dimension_numbers[2].index('C'),
                           is_input=False)

    # Reorder output spatial dimensions if the finite layer does so.
    # TODO: make more efficient / lazy.
    out_spec = tuple(c for c in dimension_numbers[2] if c not in ('N', 'C'))
    in_to_out_permutation = tuple(out_spec.index(c) for c in input_spec)
    res = res.permute_spatial(in_to_out_permutation)

    return res

  return init_fn, apply_fn, kernel_fn


@layer
def Conv(out_chan,
         filter_shape,
         strides=None,
         padding=Padding.VALID.name,
         W_std=1.0,
         W_init=_randn(1.0),
         b_std=0.0,
         b_init=_randn(1.0),
         parameterization='ntk'):
  """Layer construction function for a convolution layer.

  Based on `jax.experimental.stax.Conv`. Has a similar API apart from:

  Supports arbitrary number of spatial dimensions, with batch-first,
  channels-last layout. For different dimension numbers, use `GeneralConv`.

  `W_init` and `b_init` only change the behavior of the finite width network,
  and are not used by `kernel_fn`. In most cases, `W_std` and `b_std` should
  be used instead.

  Args:
    :padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.

    :parameterization: Either "ntk" or "standard". These parameterizations are
      the direct analogues for convolution of the corresponding
      parameterizations for `Dense` layers.
  """
  return GeneralConv(None,
                     out_chan,
                     filter_shape,
                     strides,
                     padding,
                     W_std,
                     W_init,
                     b_std,
                     b_init,
                     parameterization)


@layer
def FanOut(num):
  """Layer construction function for a fan-out layer.

  Based on `jax.experimental.stax.FanOut`.
  """
  init_fn, apply_fn = ostax.FanOut(num)
  kernel_fn = lambda kernels: [kernels] * num
  return init_fn, apply_fn, kernel_fn


@layer
def FanInSum():
  """Layer construction function for a fan-in sum layer.

  Based on `jax.experimental.stax.FanInSum`.
  """
  init_fn, apply_fn = ostax.FanInSum
  kernel_fn = lambda kernels: _fan_in_kernel_fn(kernels, None)
  return init_fn, apply_fn, kernel_fn


@layer
def FanInConcat(axis=-1):
  """Layer construction function for a fan-in concatenation layer.

  Based on `jax.experimental.stax.FanInConcat`.
  """
  init_fn, apply_fn = ostax.FanInConcat(axis)
  kernel_fn = lambda kernels: _fan_in_kernel_fn(kernels, axis)
  return init_fn, apply_fn, kernel_fn


def AvgPool(window_shape,
            strides=None,
            padding=Padding.VALID.name,
            normalize_edges=True,
            batch_axis=0,
            channel_axis=-1):
  """Layer construction function for a 2D average pooling layer.

  Based on `jax.experimental.stax.AvgPool`. Has a similar API apart from:

  Args:
    :padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.AvgPool`.

    :normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.

  """
  return _Pool(Pooling.AVG, window_shape, strides, padding, normalize_edges,
               batch_axis, channel_axis)


def SumPool(window_shape,
            strides=None,
            padding=Padding.VALID.name,
            batch_axis=0,
            channel_axis=-1):
  """Layer construction function for a 2D sum pooling layer.

  Based on `jax.experimental.stax.SumPool`. Has a similar API apart from:

  Args:
    :padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`, not
      available in `jax.experimental.stax.SumPool`.

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.
  """
  return _Pool(Pooling.SUM, window_shape, strides, padding, False,
               batch_axis, channel_axis)


@layer
def _Pool(pool_type,
          window_shape,
          strides,
          padding,
          normalize_edges,
          batch_axis,
          channel_axis):
  """Layer construction function for a 2D pooling layer.

  Based on `jax.experimental.stax.AvgPool` and `jax.experimental.stax.SumPool`.
  Has a similar API apart from:

  Args:
    :pool_type: specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`)

    :padding: in addition to `VALID` and `SAME` padding, supports `CIRCULAR`,
      not available in `jax.experimental.stax.GeneralConv`.

    :normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.

    :batch_axis: integer, batch axis.

    :channel_axis: integer, channel axis. For `kernel_fn`, channel size is
      considered to be infinite.
  """

  strides = strides or (1,) * len(window_shape)
  padding = Padding(padding)

  if pool_type == Pooling.AVG:
    pool_fn = ostax.AvgPool
  elif pool_type == Pooling.SUM:
    pool_fn = ostax.SumPool
  else:
    raise ValueError('Invalid pooling type {}'.format(pool_type))

  spec = ''.join(c for c in string.ascii_uppercase
                 if c not in ('N' ,'C'))[:len(strides)]
  for a in sorted((batch_axis, channel_axis % (2 + len(strides)))):
    if a == batch_axis:
      spec = spec[:a] + 'N' + spec[a:]
    else:
      spec = spec[:a] + 'C' + spec[a:]

  if padding == Padding.CIRCULAR:
    init_fn, _ = pool_fn(window_shape, strides, Padding.SAME.name, spec)
    _, apply_fn_0 = pool_fn(window_shape, strides, Padding.VALID.name, spec)

    def apply_fn(params, inputs, **kwargs):
      non_spatial_axes = (batch_axis, channel_axis % inputs.ndim)
      spatial_axes = tuple(i for i in range(inputs.ndim)
                           if i not in non_spatial_axes)
      inputs = _same_pad_for_filter_shape(inputs, window_shape, strides,
                                          spatial_axes, 'wrap')
      res = apply_fn_0(params, inputs, **kwargs)
      return res

  elif normalize_edges or pool_type == Pooling.SUM:
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  else:

    def rescaler(dims, strides, padding):
      del dims, strides, padding  # Unused.
      return lambda outputs, inputs, spec: outputs / np.prod(window_shape)

    pool_fn = ostax._pooling_layer(lax.add, 0., rescaler)
    init_fn, apply_fn = pool_fn(window_shape, strides, padding.name, spec)

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=False)
  def kernel_fn(kernels):
    """Kernel transformation."""
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    window_shape_kernel = window_shape[::(-1 if kernels.is_reversed else 1)]
    strides_kernel = strides[::(-1 if kernels.is_reversed else 1)]

    nngp = _pool_kernel(nngp, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges, 2)
    ntk = _pool_kernel(ntk, pool_type, window_shape_kernel, strides_kernel,
                       padding, normalize_edges, 2)
    cov1 = _pool_kernel(cov1, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges,
                        1 if kernels.diagonal_batch else 2)
    cov2 = _pool_kernel(cov2, pool_type, window_shape_kernel, strides_kernel,
                        padding, normalize_edges,
                        1 if kernels.diagonal_batch else 2)

    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk)

  return init_fn, apply_fn, kernel_fn


def GlobalSumPool(batch_axis=0, channel_axis=-1):
  """Layer construction function for a global sum pooling layer.

  Sums over and removes (`keepdims=False`) all spatial dimensions, preserving \
    the order of batch and channel axes.

  Args:
    :batch_axis: integer, batch axis.

    :channel_axis: integer, channel axis. For `kernel_fn`, channel size is \
      considered to be infinite.
  """
  return _GlobalPool(Pooling.SUM, batch_axis, channel_axis)


def GlobalAvgPool(batch_axis=0, channel_axis=-1):
  """Layer construction function for a global average pooling layer.

  Averages over and removes (`keepdims=False`) all spatial dimensions,
    preserving the order of batch and channel axes.

  Args:
    :batch_axis: integer, batch axis.

    :channel_axis: integer, channel axis. For `kernel_fn`, channel size is
      considered to be infinite.
  """
  return _GlobalPool(Pooling.AVG, batch_axis, channel_axis)


@layer
def _GlobalPool(pool_type, batch_axis, channel_axis):
  """Layer construction function for a global pooling layer.

  Pools over and removes (`keepdims=False`) all spatial dimensions, preserving
    the order of batch and channel axes.

  Args:
    :pool_type: specifies whether average or sum pooling should be performed.
      (`Pooling.AVG` or `Pooling.SUM`).
    :batch_axis: integer, batch axis.
    :channel_axis: integer, channel axis. For `kernel_fn`, channel size is
      considered to be infinite.
  """

  if pool_type == Pooling.AVG:
    pool_fn = np.mean
  elif pool_type == Pooling.SUM:
    pool_fn = np.sum
  else:
    raise ValueError('Invalid pooling type {}'.format(pool_type))

  def init_fn(rng, input_shape):
    non_spatial_axes = (batch_axis, channel_axis)
    output_shape = tuple(input_shape[i] for i in non_spatial_axes)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    non_spatial_axes = (batch_axis, channel_axis % inputs.ndim)
    spatial_axes = tuple(i for i in range(inputs.ndim)
                         if i not in non_spatial_axes)
    out = pool_fn(inputs, axis=spatial_axes)
    return out

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=False)
  def kernel_fn(kernels):
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    def _pool(ker_mat, batch_ndim):
      if not utils.is_array(ker_mat):
        return ker_mat
      spatial_axes = tuple(range(batch_ndim, ker_mat.ndim))
      return pool_fn(ker_mat, axis=spatial_axes)

    nngp = _pool(nngp, 2)
    ntk = _pool(ntk, 2)
    cov1 = _pool(cov1, 1 if kernels.diagonal_batch else 2)
    cov2 = _pool(cov2, 1 if kernels.diagonal_batch else 2)

    ndim = len(kernels.shape1)
    batch_first = batch_axis % ndim < channel_axis % ndim
    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk,
                            batch_axis=0 if batch_first else 1,
                            channel_axis=1 if batch_first else 0,
                            is_reversed=False)

  return init_fn, apply_fn, kernel_fn


@layer
def Flatten(batch_axis=0, batch_axis_out=0):
  """Layer construction function for flattening all non-batch dimensions.


  Based on `jax.experimental.stax.Flatten`, but allows to specify batch axes.

  Args:
    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.
    :batch_axis_out: integer, batch axis of the output, `0` (leading) or `1`
      (trailing).
  """
  if batch_axis_out in (0, -2):
    batch_axis_out = 0
    channel_axis_out = 1
  elif batch_axis_out in (1, -1):
    batch_axis_out = 1
    channel_axis_out = 0
  else:
    raise ValueError(
        f'`batch_axis_out` must be 0 or 1, got {batch_axis_out}.')

  def get_output_shape(input_shape):
    batch_size = input_shape[batch_axis]
    channel_size = functools.reduce(op.mul,
        input_shape[:batch_axis] + input_shape[batch_axis + 1:], 1)
    if batch_axis_out == 0:
      return (batch_size, channel_size)
    return (channel_size, batch_size)

  def init_fn(rng, input_shape):
    output_shape = get_output_shape(input_shape)
    return output_shape, ()

  def apply_fn(params, inputs, **kwargs):
    output_shape = get_output_shape(inputs.shape)
    inputs = np.moveaxis(inputs, batch_axis, batch_axis_out)
    return inputs.reshape(output_shape)

  @requires(batch_axis=batch_axis,
            channel_axis=None,
            diagonal_spatial=True)
  def kernel_fn(kernels):
    """Compute kernels."""
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    def trace(x, batch_ndim):
      if not utils.is_array(x):
        return x

      if kernels.diagonal_spatial:
        spatial_axes = tuple(range(x.ndim)[batch_ndim:])
        x = np.mean(x, spatial_axes)

      else:
        while x.ndim > batch_ndim:
          x = np.trace(x, axis1=-2, axis2=-1) / x.shape[-1]

      return x

    cov1 = trace(cov1, 1 if kernels.diagonal_batch else 2)
    cov2 = trace(cov2, 1 if kernels.diagonal_batch else 2)
    nngp = trace(nngp, 2)
    ntk = trace(ntk, 2)

    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk,
                            is_gaussian=False,
                            is_reversed=False,
                            batch_axis=batch_axis_out,
                            channel_axis=channel_axis_out,
                            diagonal_spatial=False)

  return init_fn, apply_fn, kernel_fn


@layer
def BatchNorm(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
              beta_init=initializers.zeros, gamma_init=initializers.ones):
  init_fn, apply_fn = ostax.BatchNorm(axis, epsilon, center, scale, beta_init, gamma_init)
  def kernel_fn(kernels):
    raise NotImplementedError("BatchNorm kernel not implemented")
  return init_fn, apply_fn, kernel_fn


@layer
def MaxPool(window_shape, strides=None, padding='VALID', spec=None):
  init_fn, apply_fn = ostax.MaxPool(window_shape, strides, padding, spec)
  def kernel_fn(kernels):
    raise NotImplementedError("MaxPool kernel not implemented")
  return init_fn, apply_fn, kernel_fn


@layer
def Identity():
  """Layer construction function for an identity layer.

  Based on `jax.experimental.stax.Identity`.
  """
  init_fn, apply_fn = ostax.Identity
  kernel_fn = lambda kernels: kernels
  return init_fn, apply_fn, kernel_fn


@layer
def Erf(do_backprop=False):
  return _elementwise(_erf,
                      'Erf',
                      do_backprop=do_backprop)


@layer
def Relu(do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu,
                      'ReLU',
                      a=0,
                      b=1,
                      do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@layer
def ABRelu(a, b, do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu,
                      f'ABReLU({a}, {b})',
                      a=a,
                      b=b,
                      do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@layer
def LeakyRelu(alpha, do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu,
                      f'LeakyReLU({alpha})',
                      a=alpha,
                      b=1,
                      do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@layer
def Abs(do_backprop=False, do_stabilize=False):
  return _elementwise(_ab_relu,
                      'Abs',
                      a=-1,
                      b=1,
                      do_backprop=do_backprop,
                      do_stabilize=do_stabilize)


@layer
def GlobalSelfAttention(n_chan_out,
                        n_chan_key,
                        n_chan_val,
                        n_heads,
                        fixed=True,
                        W_key_std=1.0,
                        W_value_std=1.0,
                        W_query_std=1.0,
                        W_out_std=1.0,
                        b_std=0.0,
                        W_key_init=_randn(1.0),
                        W_value_init=_randn(1.0),
                        W_query_init=_randn(1.0),
                        W_out_init=_randn(1.0),
                        b_init=_randn(1.0),
                        batch_axis=0,
                        channel_axis=-1):
  """Layer construction function for (global) scaled dot-product self-attention
  with multiple attention heads.

  Two versions of attention are available (the version to be used is
  determined by the argument `fixed`):

  1) Parametric: this is the standard scaled dot-product attention, i.e.,
   the dot product between keys and queries is scaled by the squared root
   of their dimension. The expression for `nngp`/`ntk` involves an integral
   with no known closed form and thus call to `kernel_fn` results in an error.

  2) Fixed: same as Parametric except for scaling the dot products
   between keys and queries by their dimension instead of the square root
   of the same quantity, and tying the key and query weight matrices.
   This makes the `nngp`/`ntk` analytically tractable but for the price
   that, unlike in the parametric case, the dot products of keys and queries
   converge to a constant. Because this constant would be zero
   if the key and query weights are independent, the variant where these
   two weight matrices are tied was implemented resulting in non-constant
   attention weights.

  The final computation for single head is then
   `f_h (x) + softmax(<scaling> Q(x) K(x)^T) V(x)`
  and the output of this layer is computed as
   `f(x) = concat[f_1(x) , ... , f_<n_heads> (x)] W_out + b`
  where the shape of of `b` is (n_chan_out,), i.e., single bias per channel

  The `kernel_fn` computes the limiting kernel of the outputs of this layer
  as the number of heads and the number of feature dimensions of keys/queries
  goes to infinity.

  Args:
    :n_chan_out: number of feature dimensions of outputs

    :n_chan_key: number of feature dimensions of keys/queries

    :n_chan_val: number of feature dimensions of values

    :n_heads: number of attention heads

    :fixed: if `True`, the dot products between keys and queries are
      scaled by `1 / n_chan_key` and the key and query weight matrices are tied;
      if `False`, the dot products are scaled by `1 / sqrt(n_chan_key)` and
      the key and query matrices are independent

    :W_out_std: init standard deviation of the output weights values

    :b_std: init standard deviation of the bias values

    :W_value_std: init standard deviation of the key weights values

    :W_key_std: init standard deviation of the key weights values

    :W_query_std: init standard deviation of the query weights values; if
      `fixed` is `True` (and thus key and query weights are tied---see above)
      then keys are computed with `WK = WK_std * W / sqrt(n_chan_in)` and the
      queries are computed with `WQ = W_query_std * W / sqrt(n_chan_in)` weight
      matrices

    :W_out_init: function used to sample the initial (unscaled) output weights

    :b_init:  function used to sample the initial (unscaled) biases

    :W_value_init: function used to sample the initial (unscaled) value weights

    :W_key_init: function used to sample the initial (unscaled) key weights

    :W_query_init: function used to sample the initial (unscaled) query weights
      unless `fixed` is `True` in which case the argument is ignored (see above)

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.

  Raises:
    NotImplementedError: If `fixed` is `False`, call to `kernel_fn` will result
    in an error as there is no known analytic expression for the kernel.
  """
  OV_gain = W_out_std * W_value_std
  QK_gain = W_query_std * W_key_std
  QK_prod_scaling = float(n_chan_key if fixed else n_chan_key**0.5)

  def init_fn(rng, input_shape):
    _channel_axis = channel_axis % len(input_shape)
    n_chan_in = input_shape[_channel_axis]
    output_shape = (input_shape[:_channel_axis] + (n_chan_out,) +
                    input_shape[_channel_axis + 1:])

    rng_Q, rng_K, rng_V, rng_O, rng_b = random.split(rng, 5)
    key_matrices = W_key_init(rng_K, shape=(n_heads, n_chan_in, n_chan_key))
    val_matrices = W_value_init(rng_V, shape=(n_heads, n_chan_in, n_chan_val))
    W_out = W_out_init(rng_O, shape=(n_chan_val * n_heads, n_chan_out))
    b = b_init(rng_b, shape=(n_chan_out,))

    if fixed:
      query_matrices = None
      warnings.warn('Fixed attention used -> `W_query_init` ignored, tying'
                    ' the weights (see docstring for more details).')
    else:
      query_matrices = W_query_init(rng_Q,
                                    shape=(n_heads, n_chan_in, n_chan_key))

    return output_shape, (query_matrices, key_matrices, val_matrices, W_out, b)

  def apply_fn(params, inputs, **kwargs):
    query_matrices, key_matrices, val_matrices, W_out, b = params
    n = inputs.shape[batch_axis]
    _channel_axis = channel_axis % inputs.ndim
    n_chan_in = inputs.shape[_channel_axis]
    spatial_shape = tuple(s for i, s in enumerate(inputs.shape)
                          if i not in (batch_axis, _channel_axis))

    inputs = np.moveaxis(inputs, (batch_axis, _channel_axis), (0, -1))
    inputs = inputs.reshape((n, -1, n_chan_in))

    def _inputs_dot(matrices, std):
      ret = np.dot(inputs, std * matrices / np.sqrt(n_chan_in))
      return np.moveaxis(ret, 2, 0)

    keys = _inputs_dot(key_matrices, W_key_std)
    values = _inputs_dot(val_matrices, W_value_std)
    if fixed:
      queries = keys * W_query_std / W_key_std
    else:
      queries = _inputs_dot(query_matrices, W_query_std)

    G_mat  = np.matmul(queries, np.moveaxis(keys, -1, -2))
    G_mat /= QK_prod_scaling
    G_mat = ostax.softmax(G_mat, axis=-1)

    heads = np.matmul(G_mat, values)
    heads = np.moveaxis(heads, 0, -1)
    heads = np.reshape(heads, heads.shape[:-2] + (-1,))

    ret = np.matmul(heads, W_out_std * W_out / np.sqrt(n_chan_val * n_heads))
    ret = np.reshape(ret, (n,) + spatial_shape + (n_chan_out,)) + b_std * b
    ret = np.moveaxis(ret, (0, -1), (batch_axis, _channel_axis))
    return ret

  @requires(batch_axis=batch_axis,
            channel_axis=channel_axis,
            diagonal_spatial=False)
  def kernel_fn(kernels):
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    if not fixed:
      raise NotImplementedError('No known closed form expression.')

    def _get_G_softmax(mat):
      if not kernels.diagonal_batch:
        mat = np.moveaxis(np.diagonal(mat, axis1=0, axis2=1), -1, 0)
      axes = tuple(range(mat.ndim))
      return ostax.softmax(QK_gain * mat, axis=axes[2::2])

    def _transform_kernel(mat, G1, G2=None):
      if not utils.is_array(mat):
        return mat

      G2 = G1 if G2 is None else G2

      # Spatial axes
      G1_dims = tuple(range(1, G1.ndim))
      G2_dims = tuple(range(G1.ndim, G1.ndim + G2.ndim - 1))
      mat_dims = _zip_flat(G1_dims[1::2], G2_dims[1::2])
      res_dims = _zip_flat(G1_dims[::2], G2_dims[::2])

      # Batch axes
      if mat.ndim % 2:
        G1_dims = (0,) + G1_dims
        G2_dims = (0,) + G2_dims
        mat_dims = (0,) + mat_dims
        res_dims = (0,) + res_dims
      else:
        G1_dims = (0,) + G1_dims
        G2_dims = (-1,) + G2_dims
        mat_dims = (0, -1) + mat_dims
        res_dims = (0, -1) + res_dims

      res = np.einsum(G1, G1_dims, mat, mat_dims, G2, G2_dims, res_dims,
                      optimize=True)
      return _affine(res, OV_gain, b_std)

    G1 = _get_G_softmax(cov1)
    G2 = _get_G_softmax(cov2) if cov2 is not None else G1

    cov1 = _transform_kernel(cov1, G1)
    cov2 = _transform_kernel(cov2, G2) if cov2 is not None else cov2
    nngp = _transform_kernel(nngp, G1, G2)
    ntk = (_transform_kernel(ntk, G1, G2) + 2 * (nngp - b_std**2)
           if ntk is not None else ntk)

    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk,
                            is_gaussian=True)

  return init_fn, apply_fn, kernel_fn


@layer
def LayerNorm(axis=-1, eps=1e-12, batch_axis=0, channel_axis=-1):
  """Layer normalisation.

  Args:
    :axis: integer or a tuple, specifies dimensions over which to normalize.

    :eps: float, specifies (small) positive constant to be added to the variance
      estimates in order to prevent division by zero.

    :batch_axis: integer, batch axis. Defaults to `0`, the leading axis.

    :channel_axis: integer, channel axis. Defaults to `-1`, the trailing axis.
      For `kernel_fn`, channel size is considered to be infinite.
  """
  def init_fn(rng, input_shape):
    return input_shape, ()

  def apply_fn(params, inputs, **kwargs):
    mean = np.mean(inputs, axis=axis, keepdims=True)
    var = np.var(inputs, axis=axis, keepdims=True)
    return (inputs - mean) / np.sqrt(eps + var)

  @requires(batch_axis=batch_axis, channel_axis=channel_axis)
  def kernel_fn(kernels):
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    if not kernels.is_gaussian:
      raise NotImplementedError('LayerNorm only implemented for Gaussian '
                                'inputs.')

    ndim = len(kernels.shape1)
    _channel_axis = channel_axis % ndim
    _batch_axis = batch_axis % ndim
    _axis = list(utils.canonicalize_axis(axis, kernels.shape1))

    if _channel_axis not in _axis:
      raise ValueError(f'Normalisation over channels (axis {_channel_axis})'
                       f'necessary for convergence to an asymptotic kernel; '
                       f'got axis={_axis}.')
    _axis.remove(_channel_axis)

    spatial_axes = tuple(i for i in range(len(kernels.shape1))
                         if i not in (_channel_axis, batch_axis))

    # Batch axis
    if _batch_axis in _axis:
      kernel_axis = (0,)
      _axis.remove(_batch_axis)
    else:
      kernel_axis = ()

    # Spatial axes
    kernel_axis += tuple(
        1 + spatial_axes[::(-1 if kernels.is_reversed else 1)].index(i)
        for i in _axis)

    prod11, prod12, prod22 = _get_diagonal_prod(
        eps + cov1,
        cov2 if cov2 is None else eps + cov2,
        kernels.diagonal_batch,
        kernels.diagonal_spatial,
        axis=kernel_axis)

    nngp /= np.sqrt(prod12)

    if utils.is_array(ntk):
      ntk /= np.sqrt(prod12)

    cov1 /= np.sqrt(prod11)
    if cov2 is not None:
      cov2 /= np.sqrt(prod22)

    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk)

  return init_fn, apply_fn, kernel_fn


@layer
def Dropout(rate, mode='train'):
  """Dropout layer.

  Based on `jax.experimental.stax.Dropout`. Has a similar API.

  Args:
    :rate: A float specifying the keep `rate`, e.g. `rate=1` is equivalent to
      keeping all neurons.

    :mode: either `train` or `test`.
  """
  if mode not in ('test', 'train'):
    raise ValueError('The `mode` must be either "test"  or "train".')
  if rate <= 0. or rate > 1.:
    raise ValueError('The `rate` must be > 0. and <= 1.')

  init_fn, apply_fn = ostax.Dropout(rate, mode=mode)
  kernel_fn_test = lambda kernels: kernels

  def kernel_fn_train(kernels):
    """kernel_fn for `train` mode. """
    cov1, nngp, cov2, ntk = (kernels.cov1, kernels.nngp, kernels.cov2,
                             kernels.ntk)

    if kernels.is_input:
      raise ValueError('Dropout cannot be applied to the input layer.')

    factor = 1./rate

    cov1 = _diag_mul(cov1, factor, kernels.diagonal_batch,
                     kernels.diagonal_spatial)
    cov2 = _diag_mul(cov2, factor, kernels.diagonal_batch,
                     kernels.diagonal_spatial)

    new_factor = np.where(kernels.x1_is_x2, factor, 1.)
    nngp = _diag_mul(nngp, new_factor, False, kernels.diagonal_spatial)
    ntk = _diag_mul(ntk, new_factor, False, kernels.diagonal_spatial)

    # TODO: under which condition could we leave `is_gaussian` unchanged?
    return kernels._replace(cov1=cov1,
                            nngp=nngp,
                            cov2=cov2,
                            ntk=ntk,
                            is_gaussian=False)

  kernel_fn = kernel_fn_test if mode == 'test' else kernel_fn_train

  return init_fn, apply_fn, kernel_fn


# INTERNAL UTILITIES


_CONV_KERNEL_DIMENSION_NUMBERS = ('NCHW', 'HWIO', 'NCHW')


_INPUT_REQ = 'input_req'


_DEFAULT_INPUT_REQ = frozendict.frozendict({'diagonal_batch': True,
                                            'diagonal_spatial': False,
                                            'batch_axis': 0,
                                            'channel_axis': -1})


def _get_input_req_attr(kernel_fns):
  """Gets requirements of the combined layer based on individual requirements.

  Specifically, gets the requirements to the inputs to a `serial` or `parallel`
  sequence of layers based on requirements of each layer, setting requirements
  to the most demanding among all layers.

  Args:
    :req: a 'dict` with initial requirements.
    :kernel_fns: list of 'kernel_fn`s fed to the `kernel_fns` (e.g. a list of
      convolutional layers and nonlinearities to be chained together with the
      `serial` combinator).

  Returns:
    A `dict` with combined requirements.
  """
  req = {}
  for f in reversed(kernel_fns):
    req_f = getattr(f, _INPUT_REQ, {})

    for k, v in req_f.items():
      if k in ('batch_axis', 'channel_axis'):
        req[k] = v

      elif k in ('diagonal_batch', 'diagonal_spatial'):
        # Set the most demanding marginalization requirement.
        if k in req:
          req[k] &= v
        else:
          req[k] = v

      else:
        raise NotImplementedError(k)

  return req


def _zip_flat(x, y):
  return tuple(c for xy in zip(x, y) for c in xy)


def _double_tuple(x):
  return tuple(v for v in x for _ in range(2))


def _interleave_ones(x, start_axis, end_axis, x_first):
  x_axes = x.shape[start_axis:end_axis]
  ones = (1,) * (end_axis - start_axis)
  shape = x.shape[:start_axis]
  if x_first:
    shape += _zip_flat(x_axes, ones)
  else:
    shape += _zip_flat(ones, x_axes)
  shape += x.shape[end_axis:]
  return x.reshape(shape)


def _outer_prod(x, y, start_axis, end_axis, prod_op):
  if y is None:
    y = x
  x = _interleave_ones(x, start_axis, end_axis, True)
  y = _interleave_ones(y, start_axis, end_axis, False)
  return prod_op(x, y)


def _cov_diag_batch_diag_spatial(x, batch_axis, channel_axis):
  ret = np.sum(x ** 2, axis=channel_axis)
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, new_batch_axis, 0)
  return ret


def _cov_diag_batch_full_spatial(x, batch_axis, channel_axis):
  x = np.moveaxis(x, (batch_axis, channel_axis), (0, -1))
  ret = lax.dot_general(x, x, (((x.ndim - 1,), (x.ndim - 1,)), ((0,), (0,))))
  ret = utils.zip_axes(ret, 1)
  return ret


def _cov_full_batch_full_spatial(x1, x2, batch_axis, channel_axis):
  ret = np.tensordot(x1, x2, (channel_axis, channel_axis))
  new_batch_axis = batch_axis - (1 if batch_axis > channel_axis else 0)
  ret = np.moveaxis(ret, (new_batch_axis, x1.ndim - 1 + new_batch_axis), (0, 1))
  ret = utils.zip_axes(ret, 2)
  return ret


def _cov_full_batch_diag_spatial(x1, x2, batch_axis, channel_axis):
  ret = np.matmul(np.moveaxis(x1, (batch_axis, channel_axis), (-2, -1)),
                  np.moveaxis(x2, (batch_axis, channel_axis), (-1, -2)))
  ret = np.moveaxis(ret, (-2, -1), (0, 1))
  return ret


def _cov_diagonal_batch(x, diagonal_spatial, batch_axis, channel_axis):
  if diagonal_spatial:
    ret = _cov_diag_batch_diag_spatial(x, batch_axis, channel_axis)
  else:
    ret = _cov_diag_batch_full_spatial(x, batch_axis, channel_axis)
  return ret / x.shape[channel_axis]


def _cov(x1, x2, diagonal_spatial, batch_axis, channel_axis):
  """Computes uncentred covariance (nngp) between two batches of inputs.

  Args:
    :x1: a (2+S)D (S >= 0) `np.ndarray` of shape
      `(batch_size_1, <S spatial dimensions>, n_channels)`. `batch_size_1`,
      `n_channels` may be in different positions based on `batch_axis` and
      `channel_axis`.
    :x2: an optional `np.ndarray` that has the same shape as `a` apart from
      possibly different batch (`batch_size_2`) dimension. `None` means
      `x2 == x1`.
    :diagonal_spatial: a boolean specifying whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    :batch_axis: integer, specifying which axis is the batch axis.
    :channel_axis: integer, specifying which axis is the channel / feature axis.
      For `kernel_fn`, channel size is considered to be infinite.

  Returns:
    an `np.ndarray` with uncentred batch covariance with shape
    `(batch_size_1, batch_size_2, <S spatial dimensions>)`
    if `diagonal_spatial` is `True`, or
    `(batch_size_1, batch_size_2, <2*S spatial dimensions>)`
    if `diagonal_spatial` is `False`.
  """
  x2 = x1 if x2 is None else x2

  if diagonal_spatial:
    ret = _cov_full_batch_diag_spatial(x1, x2, batch_axis, channel_axis)

  else:
    ret = _cov_full_batch_full_spatial(x1, x2, batch_axis, channel_axis)

  return ret / x1.shape[channel_axis]


def _inputs_to_kernel(x1,
                      x2,
                      diagonal_batch,
                      diagonal_spatial,
                      compute_ntk,
                      batch_axis,
                      channel_axis,
                      eps=1e-12):
  """Transforms (batches of) inputs to a `Kernel`.

  This is a private method. Docstring and example are for internal reference.

   The kernel contains the empirical covariances between different inputs and
     their entries (e.g. pixels, words, entries in a time series etc.) necessary
     to compute the covariance of the Gaussian Process corresponding to an
     infinite Bayesian or continuous gradient descent trained neural network.

   The smallest necessary number of covariance entries is tracked. For example,
     all networks are assumed to have i.i.d. weights along the channel / feature
     / logits dimensions, hence covariance between different entries along these
     dimensions is known to be 0 and is not tracked.

  Args:
    :x1: an `(S+2)`-dimensional `np.ndarray` of shape
      `(batch_size_1, height, width, depth, ..., n_channels)` with `S` spatial
      dimensions (`S >= 0`). Dimensions may be in different order based on
      `batch_axis` and `channel_axis`.
    :x2: an optional `np.ndarray` with the same shape as `x1` apart
      from possibly different batch size. `None` means `x2 == x1`.
    :diagonal_batch: a boolean specifying whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).
    :diagonal_spatial: a boolean specifying whether all (`cov1`, `ntk`, etc.)
      input covariance matrcies should store only the diagonals of the
      location-location covariances
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    :compute_ntk: a boolean, `True` to compute both NTK and NNGP kernels,
      `False` to only compute NNGP.
    :batch_axis: integer, batch axis.
    :channel_axis: integer, channel axis.  For `kernel_fn`, channel size is
      considered to be infinite.
    :eps: a small number used to check whether x1 and x2 are the same up to
        `eps`.

    Example:
      ```python
          >>> x = np.ones((10, 32, 16, 3))
          >>> o = _inputs_to_kernel(x, None,
          >>>                       diagonal_batch=True,
          >>>                       diagonal_spatial=False,
          >>>                       compute_ntk=True)
          >>> o.cov1.shape, o.ntk.shape
          (10, 32, 32, 16, 16), (10, 10, 32, 32, 16, 16)
          >>> o = _inputs_to_kernel(x, None,
          >>>                       diagonal_batch=True,
          >>>                       diagonal_spatial=True,
          >>>                       compute_ntk=True)
          >>> o.cov1.shape, o.ntk.shape
          (10, 32, 16), (10, 10, 32, 16)
          >>> x1 = np.ones((10, 128))
          >>> x2 = np.ones((20, 128))
          >>> o = _inputs_to_kernel(x1, x2,
          >>>                       diagonal_batch=True,
          >>>                       diagonal_spatial=True,
          >>>                       compute_ntk=False)
          >>> o.cov1.shape, o.nngp.shape
          (10,), (10, 20)
          >>> o.ntk
          None
      ```

  Returns:
    a `Kernel` object.
  """
  batch_axis %= x1.ndim

  if batch_axis != 0:
    # TODO: add support or clear error for batching.
    warnings.warn(f'!!! Non-leading (!= 0) batch dimension in the '
                  f'input layer is not supported for batching and empirical '
                  f'kernels, got batch_axis = {batch_axis}. !!!')

  if channel_axis is None:
    def flatten(x):
      if x is None:
        return x
      return np.moveaxis(x, batch_axis, 0).reshape((x.shape[batch_axis], -1))

    x1, x2 = flatten(x1), flatten(x2)
    batch_axis, channel_axis = 0, 1
    diagonal_spatial = False

  else:
    channel_axis %= x1.ndim

  def get_x_and_var(x):
    if x is None:
      return None, None

    if x.ndim < 2:
      raise ValueError(f'Inputs must be at least 2D (a batch dimension and a '
                       f'channel/feature dimension), got {x.ndim}.')

    # TODO: Think more about dtype automatic vs manual dtype promotion.
    x = x.astype(np.float64)

    if diagonal_batch:
      cov = _cov_diagonal_batch(x, diagonal_spatial, batch_axis, channel_axis)
    else:
      cov = _cov(x, x, diagonal_spatial, batch_axis, channel_axis)

    return x, cov

  x1, cov1 = get_x_and_var(x1)
  x2, cov2 = get_x_and_var(x2)
  nngp = _cov(x1, x2, diagonal_spatial, batch_axis, channel_axis)

  ntk = 0. if compute_ntk else None
  is_gaussian = False
  is_reversed = False
  x1_is_x2 = utils.x1_is_x2(x1, x2, eps=eps)
  is_input = False

  return Kernel(cov1,
                nngp,
                cov2,
                ntk,
                is_gaussian,
                is_reversed,
                diagonal_batch,
                diagonal_spatial,
                x1.shape,
                x2.shape if x2 is not None else x1.shape,
                x1_is_x2,
                is_input,
                batch_axis,
                channel_axis)


def _propagate_shape(init_fn, shape):
  """Statically, abstractly, evaluate the init_fn to get shape information."""
  akey = ShapedArray((2,), np.uint32)
  closed_init_fn = functools.partial(init_fn, input_shape=shape)
  args_flat, in_tree = tree_flatten(((akey,), {}))
  fun, out_tree = flatten_fun(lu.wrap_init(closed_init_fn), in_tree)
  out = pe.abstract_eval_fun(fun.call_wrapped, akey)
  out_shape = tree_unflatten(out_tree(), out)[0]
  out_shape = tree_map(lambda x: int(x.val), out_shape)
  return out_shape


def _set_shapes(init_fn, in_kernel, out_kernel):
  """Apply a kernel_fn to a Kernel propagating side information."""
  if isinstance(in_kernel, Kernel):
    shape1 = _propagate_shape(init_fn, in_kernel.shape1)
    shape2 = _propagate_shape(init_fn, in_kernel.shape2)
  elif isinstance(in_kernel, list):
    shape1 = _propagate_shape(init_fn, [k.shape1 for k in in_kernel])
    shape2 = _propagate_shape(init_fn, [k.shape2 for k in in_kernel])
  else:
    raise TypeError(f'Expected input kernel to be a `Kernel` or a list of '
                    f'`Kernel`s. Found {type(out_kernel)}.')

  if isinstance(out_kernel, Kernel):
    return out_kernel._replace(shape1=shape1, shape2=shape2)
  elif isinstance(out_kernel, list):
    return [k._replace(shape1=s1, shape2=s2) for
            k, s1, s2 in zip(out_kernel, shape1, shape2)]
  else:
    raise TypeError(f'Expected output kernel to be a `Kernel` or a list of '
                    f'`Kernel`s. Found {type(out_kernel)}.')


def _fuse_reqs(kernel_fn_reqs, default_reqs, **user_reqs):
  # Override static requirements with explicit user-specified requirements,
  # but only if they are less demanding, raise an error otherwise.
  kernel_fn_reqs = dict(kernel_fn_reqs)
  for req, v in user_reqs.items():
    if v is not None:
      if req in kernel_fn_reqs:
         if not kernel_fn_reqs[req] and v:
            raise ValueError(f'Asked to compute `kernel_fn` output with '
                             f'`{req} == {v}`, while `kernel_fn` '
                             f'requires `{req} == {kernel_fn_reqs[req]}`.')

         kernel_fn_reqs[req] |= v

      else:
        kernel_fn_reqs[req] = v

  # Fill unspecified requirements with defaults.
  for req, v in default_reqs.items():
    if req not in kernel_fn_reqs:
      kernel_fn_reqs[req] = v

  return frozendict.frozendict(kernel_fn_reqs)


def _preprocess_kernel_fn(init_fn, kernel_fn):
  """Returns a `kernel_fn` with additional arguments.

  Args:
    :init_fn: layer parameters initialization function. Used for shape
      inference.
    :kernel_fn: the `Kernel` -> `Kernel` layer propagation function.

  Returns:
    A new `kernel_fn` that does the same computation but accepts additional
      arguments to flexibly specify the required computation, and can be applied
      to either a `Kernel' or a pair of `np.ndarrray`s.
  """
  # Set empty requirements if none specified.
  if not hasattr(kernel_fn, _INPUT_REQ):
    kernel_fn = requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **user_reqs):
    out_kernel = kernel_fn(kernel, **user_reqs)
    return _set_shapes(init_fn, kernel, out_kernel)

  def kernel_fn_x1(x1, x2, get, **user_reqs):
    # Get input requirements requested by network layers, user, or defaults.
    kernel_fn_reqs = getattr(kernel_fn, _INPUT_REQ)
    reqs = _fuse_reqs(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **user_reqs)
    compute_ntk = (get is None) or ('ntk' in get)
    kernel = _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
    out_kernel = kernel_fn(kernel, **user_reqs)
    return _set_shapes(init_fn, kernel, out_kernel)

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel,
                    x2=None,
                    get=None,
                    diagonal_batch=None,
                    diagonal_spatial=None):
    """Returns the `Kernel` resulting from applying `kernel_fn` to given inputs.

    Args:
      x1_or_kernel: either a `np.ndarray` with the first batch of inputs, or a
      `Kernel`.
      x2: an optional `np.ndarray` with the second batch of inputs. `None`
        means `x2 == x1` or `x1_or_kernel is Kernel`.
      get: either `None`, a string, or a tuple of strings specifying which data
        should be returned by the kernel function. Can be "nngp", "ntk", "cov1",
        "cov2", "is_gaussian", "is_reversed", "diagonal_batch",
        "diagonal_spatial", etc.
      diagonal_batch: an optional boolean specifying whether `cov1` and
        `cov2` in all intermediary layers should store only the diagonal of the
        sample-sample covariance
        (`diagonal_batch == True`,
         `cov1.shape == (batch_size_1, ...)`),
        or the full covariance
        (`diagonal_batch == False`,
         `cov1.shape == (batch_size_1, batch_size_1, ...)`).
        Defaults to least compute-heavy setting necessary to compute the output
        `nngp` [and `ntk`] covariance.
      diagonal_spatial: an optional boolean specifying whether all (`cov1`,
        `ntk`, etc.) covariance matrcies in all intermediary layers should store
        only the diagonals of the location-location covariances
        (`diagonal_spatial == True`,
         `nngp.shape == (batch_size_1, batch_size_2, height, width, ...)`),
        or the full covariance
        (`diagonal_spatial == False`,
         `nngp.shape == (batch_size_1, batch_size_2, height, height,
                         width, width, ...)`).
        Defaults to least compute-heavy setting necessary to compute the output
        `nngp` [and `ntk`] covariance.

    Returns:
      If `get` is a string, returns the requested `np.ndarray`. If `get` is a
      tuple, returns an `AnalyticKernel` namedtuple containing only the
      requested information. If `get` is `None` then a `Kernel` object is
      returned containing all the data.
    """
    if (isinstance(x1_or_kernel, Kernel) or
        (isinstance(x1_or_kernel, list) and
         all(isinstance(k, Kernel) for k in x1_or_kernel))):
      return kernel_fn_kernel(x1_or_kernel,
                              diagonal_batch=diagonal_batch,
                              diagonal_spatial=diagonal_spatial)

    return kernel_fn_x1(x1_or_kernel, x2, get,
                        diagonal_batch=diagonal_batch,
                        diagonal_spatial=diagonal_spatial)

  setattr(kernel_fn_any, _INPUT_REQ, getattr(kernel_fn, _INPUT_REQ))
  return kernel_fn_any


def _elementwise(fn, name, **fn_kwargs):
  init_fn, apply_fn = ostax.elementwise(fn, **fn_kwargs)
  kernel_fn = lambda kernels: _transform_kernels(kernels, fn, **fn_kwargs)
  init_fn.__name__ = apply_fn.__name__ = kernel_fn.__name__ = name
  return init_fn, apply_fn, kernel_fn


def _ab_relu(x, a, b, **kwargs):
  return a * np.minimum(x, 0) + b * np.maximum(x, 0)


def _erf(x, **kwargs):
  return erf(x)


def _arccos(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arccos(x)


def _sqrt(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(x <= 0, 0, x)
  else:
    x = np.maximum(x, 0)
  return np.sqrt(x)


def _safe_sqrt(x):
  return np.sqrt(np.maximum(x, 1e-20))


def _arcsin(x, do_backprop):
  if do_backprop:
    # https://github.com/google/jax/issues/654
    x = np.where(np.abs(x) >= 1, np.sign(x), x)
  else:
    x = np.clip(x, -1, 1)
  return np.arcsin(x)


def _get_diagonal(cov, diagonal_batch, diagonal_spatial):
  """Extracts the diagonal of `cov` over all (sample, spatial) dimensions.

    Adapts computation if `cov` already stores only the diagonal along some
      dimensions based on `diagonal_batch` and `diagonal_spatial`.
  """
  if not utils.is_array(cov):
    return cov

  batch_ndim = 1 if diagonal_batch else 2
  start_axis = 2 - batch_ndim
  end_axis = batch_ndim if diagonal_spatial else -1
  cov = utils.unzip_axes(cov, start_axis, end_axis)
  return utils.diagonal_between(cov, start_axis, end_axis)


def _get_diagonal_prod(cov1,
                       cov2,
                       diagonal_batch,
                       diagonal_spatial,
                       axis=()):
  """Gets outer products of diagonals `cov1, cov1`, `cov1, cov2`, `cov2, cov2`.

  `prod11[x1, x2, h1, h2, ...]` =
   cov1[x1, [x1,], h1, [h1,], ...] * cov1[x2, [x2,], h2, [h2,], ...]`,
  `prod12[x1, x2, h1, h2, ...]` =
   cov1[x1, [x1,], h1, [h1,], ...] * cov2[x2, [x2,], h2, [h2,], ...]`,
  `prod22[x1, x2, h1, h2, ...]` =
   cov2[x1, [x1,], h1, [h1,], ...] * cov2[x2, [x2,], h2, [h2,], ...]`.

  Exact shapes of `cov1` and `cov2` are defined by `diagonal_batch` and
    `diagonal_spatial`.
  """
  axis = utils.canonicalize_axis(axis, cov1)

  cov1 = _get_diagonal(cov1, diagonal_batch, diagonal_spatial)
  cov2 = _get_diagonal(cov2, diagonal_batch, diagonal_spatial)

  cov1 = np.mean(cov1, axis=axis, keepdims=True)
  cov2 = np.mean(cov2, axis=axis, keepdims=True) if cov2 is not None else None

  end_axis = 1 if diagonal_spatial else cov1.ndim
  prod12 = _outer_prod(cov1, cov2, 0, end_axis, op.mul)

  start_axis = 1 if diagonal_batch else 0
  prod11 = _outer_prod(cov1, cov1, start_axis, end_axis, op.mul)
  prod22 = (_outer_prod(cov2, cov2, start_axis, end_axis, op.mul)
            if cov2 is not None else prod11)

  return prod11, prod12, prod22


def _get_ab_relu_kernel(ker_mat, prod, a, b, do_backprop, ntk=None):
  cosines = ker_mat / _safe_sqrt(prod)
  angles = _arccos(cosines, do_backprop)

  dot_sigma = (a**2 + b**2 - (a - b)**2 * angles / np.pi) / 2
  ker_mat = ((a - b)**2 * _sqrt(prod - ker_mat**2, do_backprop) / (2 * np.pi) +
             dot_sigma * ker_mat)

  if ntk is not None:
    ntk *= dot_sigma

  return ker_mat, ntk


def _transform_kernels_ab_relu(kernels, a, b, do_backprop, do_stabilize):
  """Compute new kernels after an `ABRelu` layer.

  See https://arxiv.org/pdf/1711.09090.pdf for the leaky ReLU derivation.
  """
  cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

  if do_stabilize:
    factor = np.max([np.max(np.abs(nngp)), 1e-12])
    nngp /= factor
    cov1 /= factor
    if cov2 is not None:
      cov2 /= factor

  prod11, prod12, prod22 = _get_diagonal_prod(cov1,
                                              cov2,
                                              kernels.diagonal_batch,
                                              kernels.diagonal_spatial)
  nngp, ntk = _get_ab_relu_kernel(nngp, prod12, a, b, do_backprop, ntk=ntk)
  if do_stabilize:
    nngp *= factor

  if kernels.diagonal_batch and kernels.diagonal_spatial:
    cov1 *= (a**2 + b**2) / 2
    if cov2 is not None:
      cov2 *= (a**2 + b**2) / 2
  else:
    cov1, _ = _get_ab_relu_kernel(cov1, prod11, a, b, do_backprop)
    if cov2 is not None:
      cov2, _ = _get_ab_relu_kernel(cov2, prod22, a, b, do_backprop)

  if do_stabilize:
    cov1 *= factor
    if cov2 is not None:
      cov2 *= factor

  return kernels._replace(cov1=cov1,
                          nngp=nngp,
                          cov2=cov2,
                          ntk=ntk,
                          is_gaussian=(a == b))


def _get_erf_kernel(ker_mat, prod, do_backprop, ntk=None):
  dot_sigma = 4 / (np.pi * np.sqrt(prod - 4 * ker_mat**2))
  ker_mat = _arcsin(2 * ker_mat / _safe_sqrt(prod), do_backprop) * 2 / np.pi

  if ntk is not None:
    ntk *= dot_sigma

  return ker_mat, ntk


def _transform_kernels_erf(kernels, do_backprop):
  """Compute new kernels after an `Erf` layer."""
  cov1, nngp, cov2, ntk = kernels.cov1, kernels.nngp, kernels.cov2, kernels.ntk

  _cov1_denom = 1 + 2 * cov1
  _cov2_denom = None if cov2 is None else 1 + 2 * cov2

  prod11, prod12, prod22 = _get_diagonal_prod(_cov1_denom,
                                              _cov2_denom,
                                              kernels.diagonal_batch,
                                              kernels.diagonal_spatial)
  nngp, ntk = _get_erf_kernel(nngp, prod12, do_backprop, ntk=ntk)

  if kernels.diagonal_batch and kernels.diagonal_spatial:
    cov1 = np.arcsin(2 * cov1 / _cov1_denom) * 2 / np.pi
    if cov2 is not None:
      cov2 = np.arcsin(2 * cov2 / _cov2_denom) * 2 / np.pi
  else:
    cov1, _ = _get_erf_kernel(cov1, prod11, do_backprop)
    if cov2 is not None:
      cov2, _ = _get_erf_kernel(cov2, prod22, do_backprop)

  return kernels._replace(cov1=cov1,
                          nngp=nngp,
                          cov2=cov2,
                          ntk=ntk,
                          is_gaussian=False)


def _transform_kernels(kernels, fn, **fn_kwargs):
  """Apply transformation to kernels.

  Args:
    kernels: a `Kernel` object.
    fn: nonlinearity function, can only be Relu, Erf or Identity.
  Returns:
    The transformed kernel.
  """
  if not kernels.is_gaussian:
    raise ValueError('An affine layer (i.e. dense or convolution) '
                     'has to be applied before a nonlinearity layer.')
  if fn is _ab_relu:
    return _transform_kernels_ab_relu(kernels, **fn_kwargs)
  if fn is _erf:
    return _transform_kernels_erf(kernels, **fn_kwargs)
  # TODO: Monte Carlo approximation to the integral (suggested by schsam.)
  raise NotImplementedError('Analaytic kernel for activiation {} is not '
                            'implmented'.format(fn))


def _affine(mat, W_std, b_std):
  """Get covariances of affine outputs if inputs have covariances `nngp`.

  The output is assumed to be `xW + b`, where `x` is the input, `W` is a matrix
    of i.i.d. Gaussian weights with std `W_std`, `b` is a vector of i.i.d.
    Gaussian biases with std `b_std`.

  Args:
    mat: a  `np.ndarray` containing sample-[sample-]position[-position]
      covariances of inputs.
    W_std: a float, standard deviation of a fully-connected layer weights.
    b_std: a float, standard deviation of a fully-connected layer biases.

  Returns:
    a 2D or 1D `np.ndarray` containing sample[-sample] [co]variances of FC
      outputs. Has the same shape as `nngp`.
  """
  if mat is None:
    return mat

  return  W_std**2 * mat + b_std**2


def _fan_in_kernel_fn(kernels, axis):
  diagonal_batch = kernels[0].diagonal_batch
  diagonal_spatial = kernels[0].diagonal_spatial

  shape1, shape2 = kernels[0].shape1, kernels[0].shape2

  ndim = len(shape1)
  axis = None if axis is None else axis % ndim
  batch_axis = kernels[0].batch_axis
  channel_axis = kernels[0].channel_axis

  # Check marginalization
  if not all(k.diagonal_batch == diagonal_batch and
             k.diagonal_spatial == diagonal_spatial and
             k.batch_axis == batch_axis and
             k.channel_axis == channel_axis
             for k in kernels):
    raise NotImplementedError('`FanIn` layers are only implemented for the '
                              'case if all input layers output the same layout '
                              'of covariance matrices, i.e. having all '
                              'matching `diagonal_batch` and '
                              '`diagonal_spatial` and other attributes.')

  # If kernels have different spatial axes order, transpose some of them.
  n_kernels = len(kernels)
  n_reversed = sum(ker.is_reversed for ker in kernels)

  if n_reversed > n_kernels / 2:
    is_reversed = True
    for i in range(n_kernels):
      if not kernels[i].is_reversed:
        kernels[i] = kernels[i].reverse()

  else:
    is_reversed = False
    for i in range(n_kernels):
      if kernels[i].is_reversed:
        kernels[i] = kernels[i].reverse()

  # Check shapes.
  if axis is None:
    if not all([k.shape1 == shape1 and k.shape2 == shape2 for k in kernels]):
      raise ValueError('All shapes should be equal in `FanInSum`.')

  else:
    new_shape1 = shape1[:axis] + shape1[axis + 1:]
    new_shape2 = shape2[:axis] + shape2[axis + 1:]
    for k in kernels:
      k_shape1 = k.shape1[:axis] + k.shape1[axis + 1:]
      k_shape2 = k.shape2[:axis] + k.shape2[axis + 1:]
      if k_shape1 != new_shape1 or k_shape2 != new_shape2:
        raise ValueError('Non-`axis` shapes should be equal in `FanInConcat`.')

  # Check if inputs are independent Gaussians.
  if axis is None or axis != channel_axis:
    is_gaussian = all(k.is_gaussian for k in kernels)
    if not is_gaussian:
      raise NotImplementedError('`FanInSum` or `FanInConcat` layer along the '
                                'non-channel axis is only implemented for the '
                                'case if all input layers guaranteed to be mean'
                                '-zero Gaussian, i.e. having all `is_gaussian '
                                'set to `True`.')
  else:
    # TODO: allow to apply nonlinearity after channelwise concatenation.
    is_gaussian = False

  # Warnings.
  warnings.warn('`FanIn` layers assume independent inputs which is not verified'
                ' in the code. Please make sure to have at least one `Dense` / '
                '`Conv` / `GlobalSenfAttention` etc. layer in each branch.')
  if axis == batch_axis:
    warnings.warn(f'Concatenation along the batch axis ({axis}) gives '
                  f'inconsistent covariances when batching - '
                  f'proceed with caution.')

  spatial_axes = tuple(i for i in range(ndim)
                  if i not in (channel_axis, batch_axis))
  # Change spatial axis according to the kernel `is_reversed`.
  if axis in spatial_axes and is_reversed:
    axis = spatial_axes[::-1][spatial_axes.index(axis)]

  # Map activation tensor axis to the covariance tensor axis.
  tensor_axis_to_kernel_axis = {
      **{
          None: None,
          batch_axis: 0,
          channel_axis: -1,
      },
      **{
          spatial_axis: idx + 1 for idx, spatial_axis in enumerate(spatial_axes)
      }
  }
  axis = tensor_axis_to_kernel_axis[axis]
  widths = [k.shape1[channel_axis] for k in kernels]

  cov1 = _concat_kernels([k.cov1 for k in kernels], axis, diagonal_batch,
                         diagonal_spatial, widths)
  cov2 = _concat_kernels([k.cov2 for k in kernels], axis, diagonal_batch,
                         diagonal_spatial, widths)
  nngp = _concat_kernels([k.nngp for k in kernels], axis,
                         False, diagonal_spatial, widths)
  ntk = _concat_kernels([k.ntk for k in kernels], axis,
                        False, diagonal_spatial, widths)
  kers = (cov1, nngp, cov2, ntk)

  return Kernel(*(
      kers + (is_gaussian,
              is_reversed,
              diagonal_batch,
              diagonal_spatial,
              None,
              None,
              kernels[0].x1_is_x2,
              kernels[0].is_input,
              batch_axis,
              channel_axis)))


def _concat_kernels(mats, axis, diagonal_batch, diagonal_spatial, widths):
  """Compute the covariance of concatenated activations with given covariances.

  Args:
    mats: a list of `np.ndarrray` covariance tensors of the same shape.
    axis: an `int` along which the covariances (not activations) are
      concatenated. `None` corresponds to sum, `-1` to averaging.
    diagonal_batch: a boolean specifying whether `cov1` and `cov2` store only
      the diagonal of the sample-sample covariance
      (`diagonal_batch == True`,
       `cov1.shape == (batch_size_1, ...)`),
      or the full covariance
      (`diagonal_batch == False`,
       `cov1.shape == (batch_size_1, batch_size_1, ...)`).
    diagonal_spatial: a boolean specifying whether only the diagonals of the
      location-location covariances will be computed,
      (`diagonal_spatial == True`,
       `nngp.shape == (batch_size_1, batch_size_2, height, width, depth, ...)`),
      or the full covariance
      (`diagonal_spatial == False`,
       `nngp.shape == (batch_size_1, batch_size_2, height, height,
                       width, width, depth, depth, ...)`).
    widths: list of integer channel widths of the finite model inputs.

  Returns:
    A new `np.ndarray` representing covariance between concatenated activations.
  """
  if mats[0] is None:
    return None

  n_mats = len(mats)
  mat_ndim = mats[0].ndim

  # Sum if `axis == None` i.e. called from `FanInSum`.
  if axis is None:
    mat = sum(mats)

  # Averaging if concatenating along features or marginalized dimension.
  elif axis == -1:
    if all(w == widths[0] for w in widths):
      widths = [1] * len(widths)
    mat = sum(mats[i] * widths[i] for i in range(n_mats)) / sum(widths)

  # Simple concatenation along the axis if the axis is not duplicated.
  elif ((axis == 0 and diagonal_batch) or
        (axis != 0 and diagonal_spatial)):
    concat_axis = axis + (0 if diagonal_batch else 1)
    mat = np.concatenate(mats, concat_axis)

  # 2D concatenation with insertion of 0-blocks if the axis is present twice.
  else:
    rows = []
    pad_axis = max(0, 2 * axis - (1 if diagonal_batch else 0))
    for i, mat in enumerate(mats):
      pads = [(0, 0)] * mat_ndim
      pads[pad_axis] = (
          sum(mats[j].shape[pad_axis] for j in range(i)),
          sum(mats[j].shape[pad_axis] for j in range(i + 1, n_mats))
      )
      rows.append(np.pad(mat, pads))
    mat = np.concatenate(rows, pad_axis + 1)

  return mat


def _same_pad_for_filter_shape(x, filter_shape, strides, axes, mode):
  """Pad an array to imitate `SAME` padding with `VALID`.

  See `Returns` section for details. This method is usually needed to implement
    `CIRCULAR` padding using `VALID` padding.

  Args:
    x: `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the convolutional spatial strides, e.g.
      e.g. `(1, 1)` for a 2D convolution.
    axes: tuple of non-negative integers, the spatial axes to apply
      convolution over (e.g. `(1, 2)` for an `NHWC` image).
    mode: a string, padding mode, for all options see
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html.

  Returns:
    A `np.ndarray` of the same dimensionality as `x` padded to a potentially
      larger shape such that a `VALID` convolution with `filter_shape` applied
      to `x` over `axes` outputs an array of the same shape as `x`.
  """
  if not utils.is_array(x):
    return x

  axes_shape = tuple(np.size(x, axis) for axis in axes)
  axes_pads = lax.padtype_to_pads(axes_shape, filter_shape, strides,
                                  Padding.SAME.name)

  pads = [(0, 0),] * x.ndim
  for i, axis in enumerate(axes):
    pads[axis] = axes_pads[i]

  x = np.pad(x, pads, mode)
  return x


def _pad_one_side(x, pads, axes, mode):
  """Pad an array on one side. See `Returns` section for details.

  Args:
    x: `np.ndarray` to pad, e.g. a 4D `NHWC` image.
    pads: tuple of integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    axes: tuple of non-negative integers, the axes to apply padding of sizes
      `pads` to.
    mode: a string, padding mode, for all options see
      https://docs.scipy.org/doc/numpy/reference/generated/numpy.pad.html.

  Returns:
    A `np.ndarray` of the same dimensionality as `x` padded to a potentially
      larger shape with `pads` applied at `axes`, where positive values in
      `pads` are applied on the left (start), and negative on the right (end).
  """
  axis_pads = [(p, 0) if p >= 0 else (0, -p) for p in pads]
  pads = [(0, 0),] * x.ndim
  for i in range(len(axes)):
    pads[axes[i]] = axis_pads[i]
  x = np.pad(x, pads, mode)
  return x


def _conv_kernel(mat, filter_shape, strides, padding, batch_ndim):
  """Compute covariance of the CNN outputs given inputs with covariance `mat`.

  Used when `kernel.diagonal_spatial == False`.

  Args:
    mat: a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of CNN inputs, where `S` is
      the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,]
        height, height, width, width, depth, depth, ...)`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim: integer, number of batch dimensions, 1 or 2.

  Returns:
    a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-[sample-]position-position covariances of CNN outputs, where `S` is
    the number of spatial dimensions (e.g. 2 for images). Has shape
    `(batch_size_1, [batch_size_2,] new_width, new_width,
      new_height, new_height, new_depth, new_depth, ...)`.
  """
  if not utils.is_array(mat):
    return mat

  if padding == Padding.CIRCULAR:
    spatial_axes = tuple(range(batch_ndim, mat.ndim))
    mat = _same_pad_for_filter_shape(
        mat,
        _double_tuple(filter_shape),
        _double_tuple(strides),
        spatial_axes,
        'wrap'
    )
    padding = Padding.VALID

  for i in range(mat.ndim - 1, batch_ndim, -2):
    spatial_i = (i - batch_ndim) // 2
    filter_i = filter_shape[spatial_i]
    stride_i = strides[spatial_i]

    ker = np.diag(np.full((filter_i,), 1. / filter_i, mat.dtype))
    for c in _CONV_KERNEL_DIMENSION_NUMBERS[1]:
      if c in ('I', 'O'):
        ker = np.expand_dims(ker, _CONV_KERNEL_DIMENSION_NUMBERS[1].index(c))

    size_i = mat.shape[i]
    mat = np.moveaxis(mat, (i - 1, i), (-2, -1))
    mat_preshape = mat.shape[:-2]
    mat = np.expand_dims(mat.reshape((-1, size_i, size_i)),
                         _CONV_KERNEL_DIMENSION_NUMBERS[0].index('C'))
    mat = lax.conv_general_dilated(
        lhs=mat,
        rhs=ker,
        window_strides=(stride_i, stride_i),
        padding=padding.name,
        dimension_numbers=_CONV_KERNEL_DIMENSION_NUMBERS)
    mat = np.squeeze(mat,
                     _CONV_KERNEL_DIMENSION_NUMBERS[2].index('C'))
    mat = mat.reshape(mat_preshape + mat.shape[-2:])

  return mat


def _conv_kernel_over_spatial(mat, filter_shape, strides, padding, batch_ndim):
  """Compute covariance of the CNN outputs given inputs with covariance `mat`.

  Used when `kernel.diagonal_spatial == True`.

  Args:
    mat: an `(S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-sample-(same position) covariances of CNN inputs. Has `batch_ndim`
      batch and `S` spatial dimensions with the shape of
      `(batch_size_1, [batch_size_2,] height, width, depth, ...)`.
    filter_shape: tuple of positive integers, the convolutional filters spatial
      shape (e.g. `(3, 3)` for a 2D convolution).
    strides: tuple of positive integers, the CNN strides (e.g. `(1, 1)` for a
      2D convolution).
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    batch_ndim: integer, number of leading batch dimensions, 1 or 2.

  Returns:
    an `(S+batch_ndim)`-dimensional `np.ndarray` containing
    sample-sample-(same position) covariances of CNN outputs. Has `batch_ndim`
    batch and `S` spatial dimensions with the shape of
    `(batch_size_1, [batch_size_2,] new_height, new_width, new_depth, ...)`.
  """
  if not utils.is_array(mat):
    return mat

  spatial_axes = tuple(range(mat.ndim)[batch_ndim:])

  if padding == Padding.CIRCULAR:
    mat = _same_pad_for_filter_shape(mat, filter_shape, strides,
                                     spatial_axes, 'wrap')
    padding = Padding.VALID

  ker = np.full((1, 1) + filter_shape, 1. / np.prod(filter_shape), mat.dtype)

  batch_shape, spatial_shape = mat.shape[:batch_ndim], mat.shape[batch_ndim:]
  mat = np.reshape(mat, (-1,) + spatial_shape)
  mat = np.expand_dims(mat, 1)
  mat = lax.conv_general_dilated(mat, ker, strides, padding.name)
  mat = mat.reshape(batch_shape + mat.shape[2:])
  return mat


def _pool_kernel(mat,
                 pool_type,
                 window_shape,
                 strides,
                 padding,
                 normalize_edges,
                 batch_ndim):
  """Get covariances of pooling outputs given inputs covariances `mat`.

  Args:
    mat: a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of pooling inputs, where `S`
      is the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,]
        height, height, width, width, depth, depth, ...)`.
    pool_type: a `Pooling` enum, e.g. `Pooling.AVG`.
    window_shape: tuple of two positive integers, the pooling spatial shape
      (e.g. `(3, 3)`).
    strides: tuple of two positive integers, the pooling strides, e.g. `(1, 1)`.
    padding: a `Padding` enum, e.g. `Padding.CIRCULAR`.
    normalize_edges: `True` to normalize output by the effective receptive
      field, `False` to normalize by the window size. Only has effect at the
      edges when `SAME` padding is used. Set to `True` to retain correspondence
      to `ostax.AvgPool`.
    batch_ndim: integer, number of leading batch dimensions, 1 or 2.

  Returns:
      a `(2*S+batch_ndim)`-dimensional `np.ndarray` containing
      sample-[sample-]position-position covariances of pooling outputs, where
      `S` is the number of spatial dimensions (e.g. 2 for images). Has shape
      `(batch_size_1, [batch_size_2,]
        height, height, width, width, depth, depth, ...)`.
  """
  if not utils.is_array(mat):
    return mat

  if padding == Padding.CIRCULAR:
    spatial_axes = tuple(range(batch_ndim, mat.ndim))
    mat = _same_pad_for_filter_shape(mat, _double_tuple(window_shape),
                                     _double_tuple(strides), spatial_axes,
                                     'wrap')
    padding = Padding.VALID

  window_shape = (1,) * batch_ndim + _double_tuple(window_shape)
  strides = (1,) * batch_ndim + _double_tuple(strides)

  nngp_out = lax.reduce_window(mat, 0., lax.add, window_shape, strides,
                               padding.name)

  if pool_type == Pooling.AVG:
    if padding == Padding.SAME and normalize_edges:
      # `SAME` padding in `jax.experimental.stax.AvgPool` normalizes by actual
      # window size, which is smaller at the edges.
      one = np.ones(mat.shape, mat.dtype)
      window_sizes = lax.reduce_window(one, 0., lax.add, window_shape, strides,
                                       padding.name)
      nngp_out /= window_sizes
    else:
      nngp_out /= np.prod(window_shape)

  return nngp_out


def _diag_mul_no_marg(x, factor, diagonal_batch):
  diag = x

  if diagonal_batch:
    idx = (slice(None),)
    batch_ndim = 1
  else:
    if x.shape[0] != x.shape[1]:
      return x
    idx = ()
    batch_ndim = 2

  ndims = x.ndim // 2
  for i in range(ndims):
    shape = [1] * ndims
    size = x.shape[2 - batch_ndim + 2 * i]
    shape[i] = size
    idx += (np.arange(size).reshape(shape),) * 2
    diag = np.diagonal(diag, axis1=2 - batch_ndim, axis2=3 - batch_ndim)

  x = ops.index_update(x, idx, diag * factor)
  return x


def _diag_mul_over_spatial(x, factor, diagonal_batch):
  if diagonal_batch:
    x = factor * x

  else:
    if x.shape[0] != x.shape[1]:
      return x
    idx = np.diag_indices(x.shape[0]) + (Ellipsis,)
    diag = np.moveaxis(np.diagonal(x), -1, 0)
    x = ops.index_update(x, idx, diag * factor)

  return x


def _diag_mul(x, factor, diagonal_batch, diagonal_spatial):
  if not utils.is_array(x):
    return x

  if diagonal_spatial:
    return _diag_mul_over_spatial(x, factor, diagonal_batch)

  return _diag_mul_no_marg(x, factor, diagonal_batch)
