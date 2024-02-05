import sys

import numpy as np
from numpy.typing import NDArray

try:
    from typing import TypeAlias
except ImportError:
    assert sys.version_info >= (3, 9), "Python 3.9 is required"


NDarray32: TypeAlias = NDArray[np.float32]


def compute_analytical_solution(
    X: NDarray32, Y: NDarray32, Lx: float, Ly: float
) -> NDarray32:
    return np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)


def add_boundary_conditions(
    solution: NDarray32,
    X: NDarray32,
    Y: NDarray32,
    Lx: float = 0.01,
    Ly: float = 0.01,
) -> None:
    solution = np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly)
    solution[1:-2, 1:-2] = 0.0


def laplacian_operator(
    array: NDarray32, dx: float = 0.01, dy: float = 0.01
) -> NDarray32:
    return (
        -2.0 * array[1:-1, 1:-1] + array[1:-1, :-2] + array[1:-1, 2:]
    ) / dx / dx + (
        -2.0 * array[1:-1, 1:-1] + array[:-2, 1:-1] + array[2:, 1:-1]
    ) / dy / dy


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
    operator_dot_direction: NDarray32 = np.zeros_like(solution)
    compute_residual(residual, solution, rhs)  # initial residual
    direction: NDarray32 = residual.copy()

    history: list[np.float32] = []
    diff: np.float32 = np.float32(1e8)
    i: int = 0
    while diff > relative_tol and i < max_iter:
        solution_k: NDarray32 = solution.copy()
        residual_k: NDarray32 = residual.copy()
        compute_laplacian_on_array(operator_dot_direction, direction)
        alpha = np.sum(residual * residual) / np.sum(
            direction * operator_dot_direction
        )
        # update solution and residual
        solution = solution_k + alpha * direction
        residual = residual_k - alpha * operator_dot_direction
        # update descent direction
        beta = np.sum(residual * residual) / np.sum(residual_k * residual_k)
        direction = residual + beta * direction
        # compute relative L2-norm of the difference
        diff = compute_norm_to_reference(solution, solution_k)
        history.append(diff)
        i += 1

    return solution, i, history


def compute_absolute_error_difference(
    solution: NDarray32, reference: NDarray32
) -> np.float32:
    return np.sqrt(np.sum((solution - reference) ** 2))


def compute_norm_to_reference(
    solution: NDarray32,
    reference: NDarray32,
    atol: float = 1e-12,
) -> np.float32:
    l2_difference = np.sqrt(np.sum((solution - reference) ** 2))
    l2_reference = np.sqrt(np.sum(reference**2))

    if l2_difference < atol or l2_reference < atol:
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
        -(1 / Lx / Lx + 1 / Ly / Ly)
        * np.pi**2
        * np.sin(np.pi * X / Lx)
        * np.cos(np.pi * Y / Ly)
    )
    solution = np.zeros_like(X)
    add_boundary_conditions(solution, X, Y, Lx, Ly)

    return X, Y, rhs, solution


def test_conjugate_gradient() -> None:
    Lx, Ly = 1.0, 1.0
    X, Y, rhs, solution = create_mesh_and_rhs(0.01, 0.01, Lx, Ly)
    max_iter, rtol = 1000, 1e-8
    solution, iterations, history = ConjugateGradientDescent(
        solution, rhs, max_iter, rtol
    )
    analytical_solution = compute_analytical_solution(X, Y, Lx, Ly)
    error_to_reference = compute_absolute_error_difference(
        solution, analytical_solution
    )
    print(
        f"CG converged in {iterations}\n"
        f"with relative tolerance {history[-1]}\n"
        f"Compute norm to reference {error_to_reference}\n\n"
    )


def test_gradient_descent() -> None:
    Lx, Ly = 1.0, 1.0
    X, Y, rhs, solution = create_mesh_and_rhs(0.01, 0.01, Lx, Ly)
    max_iter, rtol = 10000, 1e-8

    solution, iterations, history = SteppestDescent(
        solution, rhs, max_iter, rtol
    )
    analytical_solution = compute_analytical_solution(X, Y, Lx, Ly)
    error_to_reference = compute_absolute_error_difference(
        solution, analytical_solution
    )
    print(
        f"Gradient Descent converged in {iterations}\n"
        f"with relative tolerance {history[-1]}\n"
        f"Compute norm to reference {error_to_reference}\n\n"
    )


if __name__ == "__main__":
    test_gradient_descent()
    test_conjugate_gradient()
