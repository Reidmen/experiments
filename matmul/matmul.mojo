import benchmark
from memory import memset_zero
from random import rand, random_float64
from algorithm import vectorize

alias float32 = DType.float32

alias M = 1024
alias N = 1024
alias K = 1024

struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[float32]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[float32].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[float32]):
        self.data = data

    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[float32].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[float32]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[float32]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[float32, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[float32, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

fn matmul_naive(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

# simdwidthof = number of float32 elements that fit into a single SIMD register
# using a 2x multiplier allows some SIMD operations to run in the same cycle
alias nelts = simdwidthof[DType.float32]() * 2

fn matmul_nelts(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols - nelts + 1, nelts):
                C.store(m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv))

            # Handle remaining elements with scalars.
            for n in range(nelts * (C.cols // nelts), C.cols):
                C[m, n] += A[m, k] * B[k, n]

fn matmul_vect(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            @parameter
            fn dot[nelts: Int](n: Int):
                C.store(m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n))
            vectorize[dot, nelts, size = C.cols]()

@always_inline
fn benchmark_matmultyped[
    func: fn (Matrix, Matrix, Matrix) -> None]() -> Float32:
    var C = Matrix[M, N]()
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=1).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops: Float32 = ((2 * M * N * K) / secs) / 1e9
    print(gflops, "GFLOP/s")

    return gflops

def main():
    var gflops_matmul = benchmark_matmultyped[matmul_naive]()
    var gflops_matmul_nelts = benchmark_matmultyped[matmul_nelts]()
    var gflops_matmul_vect = benchmark_matmultyped[matmul_vect]() 

    var speedup: Float32 = gflops_matmul_nelts / gflops_matmul
    print("Using SIMD intructions", speedup, "x speedup over naive approach")

    var speedup_vect: Float32 = gflops_matmul_vect / gflops_matmul
    print("Using SIMD intructions and vectorized", speedup_vect, "x speedup over naive approach")


