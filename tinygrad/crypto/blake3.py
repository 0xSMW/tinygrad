from __future__ import annotations
from tinygrad import Tensor, dtypes
import numpy as np

# Minimal BLAKE3-like tree hash implemented with tinygrad kernels. Each 64-byte
# message chunk is compressed in parallel, then chaining values are merged
# hierarchically using the same compression function. The design mirrors the
# official algorithm, though without keyed or extendable output modes.

IV = Tensor([0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
             0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19], dtype=dtypes.uint32)

SIGMA = [
  [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
  [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8],
  [3,4,10,12,13,2,7,14,11,6,5,1,9,8,15,0],
  [10,7,12,9,14,3,13,15,6,2,0,5,1,11,8,4],
  [7,13,9,3,1,12,11,14,2,6,5,10,4,0,15,8],
  [9,0,5,7,2,4,10,15,14,1,11,12,6,8,3,13],
  [2,6,9,10,0,12,11,8,3,4,13,15,1,14,5,7],
]

CHUNK_LEN = 64  # bytes
FLAGS_CHUNK_START = 1
FLAGS_CHUNK_END = 2
FLAGS_PARENT = 4
FLAGS_ROOT = 8

MASK = Tensor([0xffffffff], dtype=dtypes.uint32)


def _rotr(x:Tensor, k:int) -> Tensor:
  return ((x >> k) | (x << (32 - k))) & MASK


def _g(a:Tensor, b:Tensor, c:Tensor, d:Tensor, x:Tensor, y:Tensor):
  a = (a + b + x) & MASK
  d = _rotr(d ^ a, 16)
  c = (c + d) & MASK
  b = _rotr(b ^ c, 12)
  a = (a + b + y) & MASK
  d = _rotr(d ^ a, 8)
  c = (c + d) & MASK
  b = _rotr(b ^ c, 7)
  return a, b, c, d


def compress_kernel(blocks:Tensor, cv:Tensor, counter:Tensor, flags:Tensor) -> Tensor:
  """Compress 64-byte blocks in parallel. blocks shape (N,16) uint32."""
  N = blocks.shape[0]
  v = [cv[:,i] for i in range(8)] + [IV[i].expand(N) for i in range(4)] + [counter & 0xffffffff,
       counter >> 32, Tensor.full((N,), 64, dtype=dtypes.uint32), flags]
  for r in range(7):
    s = SIGMA[r]
    v[0],v[4],v[8],v[12] = _g(v[0],v[4],v[8],v[12], blocks[:,s[0]], blocks[:,s[1]])
    v[1],v[5],v[9],v[13] = _g(v[1],v[5],v[9],v[13], blocks[:,s[2]], blocks[:,s[3]])
    v[2],v[6],v[10],v[14] = _g(v[2],v[6],v[10],v[14], blocks[:,s[4]], blocks[:,s[5]])
    v[3],v[7],v[11],v[15] = _g(v[3],v[7],v[11],v[15], blocks[:,s[6]], blocks[:,s[7]])
    v[0],v[5],v[10],v[15] = _g(v[0],v[5],v[10],v[15], blocks[:,s[8]], blocks[:,s[9]])
    v[1],v[6],v[11],v[12] = _g(v[1],v[6],v[11],v[12], blocks[:,s[10]], blocks[:,s[11]])
    v[2],v[7],v[8],v[13] = _g(v[2],v[7],v[8],v[13], blocks[:,s[12]], blocks[:,s[13]])
    v[3],v[4],v[9],v[14] = _g(v[3],v[4],v[9],v[14], blocks[:,s[14]], blocks[:,s[15]])
  out = [(v[i] ^ v[i+8]) for i in range(8)]
  return Tensor.stack(*out, dim=1)


def blake3_hash(data:bytes, device:str|None=None) -> bytes:
  """Return a 32-byte digest computed with a tree of compression kernels."""
  device = device or "REMOTE"
  arr = np.frombuffer(data, dtype=np.uint8)
  pad = (-len(arr)) % CHUNK_LEN
  if pad:
    arr = np.concatenate([arr, np.zeros(pad, np.uint8)])
  blocks = arr.view(np.uint32).reshape(-1, 16)

  # parallel leaf compression
  t = Tensor(blocks, dtype=dtypes.uint32, device=device)
  cv = IV.reshape(1, 8).expand(t.shape[0], 8)
  flags = Tensor.full((t.shape[0],), FLAGS_CHUNK_START | FLAGS_CHUNK_END, dtype=dtypes.uint32, device=device)
  counter = Tensor.arange(t.shape[0], dtype=dtypes.uint64, device=device)
  cvs = compress_kernel(t, cv, counter, flags)

  # hierarchical reduction tree
  while cvs.shape[0] > 1:
    if cvs.shape[0] % 2 == 1:
      cvs = Tensor.cat(cvs, cvs[-1].reshape(1, 8), dim=0)
    pair_blocks = Tensor.stack(cvs[0::2], cvs[1::2], dim=1).reshape(-1, 16)
    flag_val = FLAGS_PARENT | (FLAGS_ROOT if pair_blocks.shape[0] == 1 else 0)
    flags = Tensor.full((pair_blocks.shape[0],), flag_val, dtype=dtypes.uint32, device=device)
    cvs = compress_kernel(pair_blocks,
                          IV.reshape(1, 8).expand(pair_blocks.shape[0], 8),
                          Tensor.zeros(pair_blocks.shape[0], dtype=dtypes.uint64, device=device),
                          flags)

  return bytes(cvs.reshape(-1).cpu().numpy().astype(np.uint32).tobytes())

