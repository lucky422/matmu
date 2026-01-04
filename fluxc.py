#!/usr/bin/env python3
import ast
import argparse
import subprocess
import tempfile
from pathlib import Path

# -----------------------------
# Phase 1 Frontend: Python AST → find `return a @ b`
# -----------------------------
class FluxFrontend(ast.NodeVisitor):
    def __init__(self):
        self.fn_name = None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Look for: return <something> @ <something>
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                v = stmt.value
                if isinstance(v, ast.BinOp) and isinstance(v.op, ast.MatMult):
                    self.fn_name = node.name
                    return
        self.generic_visit(node)

def parse_matmul_fn(py_src: str) -> str:
    tree = ast.parse(py_src)
    fe = FluxFrontend()
    fe.visit(tree)
    if not fe.fn_name:
        raise SystemExit("Phase 1 only supports a function containing: `return a @ b`")
    return fe.fn_name

# -----------------------------
# Phase 1 Backend: emit MLIR with linalg.matmul on memrefs
# -----------------------------
def emit_mlir_matmul(fn_name: str, M: int, K: int, N: int) -> str:
    # memref-based linalg.matmul + llvm.emit_c_interface
    # so the function uses C ABI-compatible memref descriptors.
    return f"""\
module {{
  func.func @{fn_name}(%A: memref<{M}x{K}xf32>, %B: memref<{K}x{N}xf32>, %C: memref<{M}x{N}xf32>) attributes {{ llvm.emit_c_interface }} {{
    linalg.matmul ins(%A, %B : memref<{M}x{K}xf32>, memref<{K}x{N}xf32>) outs(%C : memref<{M}x{N}xf32>)
    return
  }}
}}
"""

def run(cmd):
    subprocess.check_call(cmd)

def build_cpu_so(py_file: Path, out_so: Path, M: int, K: int, N: int, fn_override: str | None):
    py_src = py_file.read_text(encoding="utf-8")
    fn_name = fn_override if fn_override else parse_matmul_fn(py_src)

    mlir_text = emit_mlir_matmul(fn_name, M, K, N)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_mlir  = td / "kernel_in.mlir"
        opt_mlir = td / "kernel_opt.mlir"
        llvm_ll  = td / "kernel.ll"
        obj_o    = td / "kernel.o"

        in_mlir.write_text(mlir_text, encoding="utf-8")

        # Lower: linalg -> loops -> LLVM dialect, then translate to LLVM IR
        run([
            "mlir-opt",
            str(in_mlir),
        
            # Linalg → loops (SCF)
            "--linalg-generalize-named-ops",
            "--convert-linalg-to-loops",
        
            # Canonical lowering
            "--lower-affine",
        
            # SCF → CF (your build supports this)
            "--convert-scf-to-cf",
        
            # CF → LLVM (your build supports this — critical!)
            "--convert-cf-to-llvm",
        
            # Arithmetic/math → LLVM
            "--convert-arith-to-llvm",
            "--convert-math-to-llvm",
        
            # Memref lowering helpers (needed for descriptors + strides)
            "--expand-strided-metadata",
            "--finalize-memref-to-llvm",
        
            # Func → LLVM + cleanup
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
        
            "-o", str(opt_mlir),
        ])



        run([
            "mlir-translate",
            str(opt_mlir),
            "--mlir-to-llvmir",
            "-o", str(llvm_ll),
        ])

        # Build shared library
        run(["clang", "-O3", "-fPIC", "-c", str(llvm_ll), "-o", str(obj_o)])
        run(["clang", "-shared", str(obj_o), "-o", str(out_so)])

    print(f"[ok] built {out_so}  (fn={fn_name}, A={M}x{K}, B={K}x{N}, C={M}x{N})")

def main():
    ap = argparse.ArgumentParser(description="Flux Phase 1 CPU: Python matmul -> MLIR -> cpu.so")
    ap.add_argument("py_file", type=Path, help="Python file with: def matmul(a,b): return a @ b")
    ap.add_argument("--out", type=Path, default=Path("cpu.so"))
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--K", type=int, default=256)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--fn", type=str, default=None, help="Override function name (optional)")
    args = ap.parse_args()

    build_cpu_so(args.py_file, args.out, args.M, args.K, args.N, args.fn)

if __name__ == "__main__":
    main()
