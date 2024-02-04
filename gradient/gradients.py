from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias

NDarray16: TypeAlias = NDArray[np.float16]
NDarray32: TypeAlias = NDArray[np.float32]
NDarray64: TypeAlias = NDArray[np.float64]


def laplacian_operator(
    array: NDarray32, dx: float = 0.01, dy: float = 0.01
) -> NDarray32:
    return (
        (
            -4.0 * array[1:-1, 1:-1]
            + array[1:-1, :-2]
            + array[1:-1, 2:]
            + array[:2, 1:-1]
            + array[2:, 1:-1]
        )
        / dx
        / dy
    )


def compute_residual(
    residual: NDarray32, array: NDarray32, rhs: NDarray32
) -> None:
    residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - laplacian_operator(array)


def compute_laplacian_on_array(
    operator_holder: NDarray32, array: NDarray32
) -> None:
    operator_holder[1:-1, 1:-1] = laplacian_operator(array)


def SteppestDescent(
    array: NDarray32,
    rhs: NDarray32,
    max_iter: int,
    relative_tol: float = 1e-6,
) -> tuple[NDarray32, int, list[np.float32]]:
    solution: NDarray32 = array.copy()
    residual: NDarray32 = np.zeros_like(solution)
    operator_on_residual: NDarray32 = np.zeros_like(solution)

    history: list[np.float32] = []
    diff: np.float32 = np.float32(1e8)
    i: int = 0
    while diff > relative_tol and i < max_iter:
        solution_k = solution.copy()
        compute_residual(residual, solution, rhs)
        compute_laplacian_on_array(operator_on_residual, residual)
        alpha = np.sum(residual * residual) / np.sum(
            residual * operator_on_residual
        )
        solution = solution_k + alpha * residual
        diff = compute_norm_to_reference(solution, solution_k)
        history.append(diff)
        i += 1

    return solution, i, history


def ConjugateGradientDescent(
    array: NDarray32,
    rhs: NDarray32,
    max_iter: int,
    relative_tol: float = 1e-6,
) -> tuple[NDarray32, int, list[np.float32]]:
    solution: NDarray32 = array.copy()
    residual: NDarray32 = np.zeros_like(solution)
    direction: NDarray32 = np.zeros_like(solution)
    operator_dot_direction: NDarray32 = np.zeros_like(solution)
    compute_residual(residual, solution, rhs)  # initial residual

    history: list[np.float32] = []
    diff: np.float32 = np.float32(1e8)
    i: int = 0
    while diff > relative_tol and i < max_iter:
        solution_k = solution.copy()
        residual_k: NDarray32 = residual.copy()
        compute_laplacian_on_array(operator_dot_direction, direction)
        alpha = np.dot(residual, residual) / np.dot(
            direction, operator_dot_direction
        )
        # update solution and residual
        solution = solution_k + alpha * residual
        residual = residual_k - alpha * operator_dot_direction
        # update descent direction
        beta = np.sum(residual * residual) / np.sum(residual_k * residual_k)
        direction = residual + beta * direction
        # compute relative L2-norm of the difference
        diff = compute_norm_to_reference(solution, solution_k)
        history.append(diff)
        i += 1

    return solution, i, history


def compute_norm_to_reference(
    solution: NDarray32,
    reference: NDarray32,
    atol: float = 1e-12,
) -> np.float32:
    l2_difference = np.sqrt(np.sum((solution - reference) ** 2))
    l2_reference = np.sqrt(np.sum(reference**2))

    if l2_difference < atol:
        return l2_difference
    else:
        return l2_difference / l2_reference


def create_mesh_and_rhs(
    dx: float, dy: float, Lx: float, Ly: float
) -> tuple[NDarray32, NDarray32, NDarray32, NDarray32]:
    x = np.arange(0.0, Lx, step=dx)
    y = np.arange(0.0, Ly, step=dy)
    X, Y = np.meshgrid(x, y)
    rhs = (
        (2.0 * np.pi / Lx)
        * (np.pi / Ly)
        * np.sin(np.pi * X / Lx)
        * np.cos(np.pi * Y / Ly)
    )
    solution = np.zeros_like(X)

    return X, Y, rhs, solution


def test_conjugate_gradient() -> None:
    X, Y, rhs, solution = create_mesh_and_rhs(0.01, 0.01, 1.0, 1.0)
    max_iter, rtol = 1000, 1e-8

    solution, iterations, history = ConjugateGradientDescent(
        solution, rhs, max_iter, rtol
    )
    print(f"CG converged in {iterations} "
          f"with relative tolerance {history[-1]}")


def test_gradient_descent() -> None:
    X, Y, rhs, solution = create_mesh_and_rhs(0.01, 0.01, 1.0, 1.0)
    max_iter, rtol = 1000, 1e-8

    solution, iterations, history = ConjugateGradientDescent(
        solution, rhs, max_iter, rtol
    )
    print(
        f"Gradient Descent converged in {iterations} "
        f"with relative tolerance {history[-1]}"
    )


if __name__ == "__main__":
    test_gradient_descent()
    test_conjugate_gradient()
