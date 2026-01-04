import ctypes
import numpy as np

# MLIR C-interface memref descriptor for rank-2:
# { allocated, aligned, offset, sizes[2], strides[2] }
class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_longlong),
        ("sizes", ctypes.c_longlong * 2),
        ("strides", ctypes.c_longlong * 2),
    ]

def as_memref2d(arr: np.ndarray) -> MemRef2D:
    if arr.dtype != np.float32:
        raise ValueError("Expected float32")
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Expected C-contiguous array")
    m, n = arr.shape
    ptr = arr.ctypes.data
    # row-major contiguous: stride0=n, stride1=1 (in elements)
    return MemRef2D(
        allocated=ptr,
        aligned=ptr,
        offset=0,
        sizes=(ctypes.c_longlong * 2)(m, n),
        strides=(ctypes.c_longlong * 2)(n, 1),
    )

lib = ctypes.CDLL("./cpu.so")

# symbol name = function name in matmul_src.py
fn = lib.matmul
fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef2D)]
fn.restype = None

M, K, N = 128, 256, 128
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

Aref = as_memref2d(A)
Bref = as_memref2d(B)
Cref = as_memref2d(C)

fn(ctypes.byref(Aref), ctypes.byref(Bref), ctypes.byref(Cref))

ref = A @ B
print("max abs error:", float(np.max(np.abs(C - ref))))
