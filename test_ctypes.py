import ctypes
import numpy as np

# Ranked memref descriptor (LLVM lowering) for rank-2:
# { allocated, aligned, offset, sizes[2], strides[2] }
class MemRef2D(ctypes.Structure):
    _fields_ = [
        ("allocated", ctypes.c_void_p),
        ("aligned", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64 * 2),
        ("strides", ctypes.c_int64 * 2),
    ]

def as_memref2d(arr: np.ndarray) -> MemRef2D:
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    arr = np.ascontiguousarray(arr)
    m, n = arr.shape
    ptr = arr.ctypes.data
    return MemRef2D(
        allocated=ptr,
        aligned=ptr,
        offset=0,
        sizes=(ctypes.c_int64 * 2)(m, n),
        strides=(ctypes.c_int64 * 2)(n, 1),  # row-major contiguous
    )

lib = ctypes.CDLL("./cpu.so")

# Prefer the C-interface wrapper if present (this is the correct ABI)
if hasattr(lib, "_mlir_ciface_matmul"):
    fn = getattr(lib, "_mlir_ciface_matmul")
    fn.argtypes = [ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef2D), ctypes.POINTER(MemRef2D)]
    fn.restype = None
    call_style = "ciface (pointer to memref desc)"
    use_byref = True
elif hasattr(lib, "matmul"):
    # Fallback: some builds export the raw func expecting memref desc BY VALUE
    fn = getattr(lib, "matmul")
    fn.argtypes = [MemRef2D, MemRef2D, MemRef2D]
    fn.restype = None
    call_style = "raw (memref desc by value)"
    use_byref = False
else:
    raise RuntimeError("No matmul symbol found. Run: nm -D cpu.so | grep -i matmul")

print("Using:", call_style)

M, K, N = 128, 256, 128
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)
C = np.zeros((M, N), dtype=np.float32)

Aref = as_memref2d(A)
Bref = as_memref2d(B)
Cref = as_memref2d(C)

if use_byref:
    fn(ctypes.byref(Aref), ctypes.byref(Bref), ctypes.byref(Cref))
else:
    fn(Aref, Bref, Cref)

ref = A @ B
print("max abs error:", float(np.max(np.abs(C - ref))))
