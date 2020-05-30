from jax import lax
import jax.numpy as np
import numpy as onp


def gather_rolled_idx(var1, start_indices, fillvalue=0.):
  """
  var1: B,H,W,...
f start_indices: A,B,3
  output: A,B,H,W,...
  """
  widths = [(0, 0, 0)] * len(var1.shape)
  widths[1] = (0, var1.shape[1], 0)
  widths[2] = (0, var1.shape[2], 0)
  pad_var1 = lax.pad(var1, np.asarray(fillvalue, dtype=var1.dtype), widths)

  slice_sizes = (1, *var1.shape[1:])

  return np.squeeze(lax.gather(
      pad_var1, start_indices, lax.GatherDimensionNumbers(
          offset_dims=tuple(range(2, 2+len(var1.shape))),
          collapsed_slice_dims=(),
          start_index_map=(0, 1, 2)),
      slice_sizes), axis=2)

if __name__ == '__main__':
    var1 = np.arange(1, 10).reshape((3, 3)) + 10*np.arange(3)[:, None, None]
    var1 = var1.astype(np.float32)

    idx = np.array([[
        [0, 0, 0],
        [1, 1, 2],
        [2, 0, 1],
    ]])

    out = gather_rolled_idx(var1, idx)
    expected_out = np.array([[
            [[1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]],
            [[16, 0, 0],
            [19, 0, 0],
            [0, 0, 0]],
            [[22, 23, 0],
            [25, 26, 0],
            [28, 29, 0]]
        ]], dtype=np.float32)
    assert out.shape == (*idx.shape[:2], *var1.shape[1:])
    assert np.allclose(out, expected_out)

    var1_inp = np.stack([var1, var1+0.2, var1+0.4], axis=3)
    out = gather_rolled_idx(var1_inp, idx)
    expected_out = onp.stack([expected_out, expected_out+0.2, expected_out+0.4], -1)
    expected_out[expected_out < 1]  = 0

    assert out.shape == (*idx.shape[:2], *var1_inp.shape[1:])
    assert np.allclose(out, expected_out)
