import pathlib
import sys

import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot, cm
from mpl_toolkits import mplot3d

try:
    from typing import TypeAlias
except ImportError:
    assert sys.version_info >= (3, 9), "Python 3.9 is required"


NDarray64: TypeAlias = NDArray[np.float64]


def compute_analytical_solution(
    X: NDarray64, Y: NDarray64, Lx: float, Ly: float
) -> NDarray64:
    """Computes analytical_solution solution used as reference."""
    return np.array(
        np.sin(np.pi * X / Lx) * np.cos(np.pi * Y / Ly), dtype=np.float64
    )


def add_boundary_conditions(
    solution: NDarray64,
    X: NDarray64,
    Y: NDarray64,
    Lx: float = 0.01,
    Ly: float = 0.01,
) -> None:
    """Imposes dirichlet boundary conditions in solution on
    solution array"""
    solution[0, :] = np.sin(np.pi * X[0, :] / Lx)  # (y, x) for solution array
    solution[-1, :] = -np.sin(np.pi * X[-1, :] / Lx)


def laplacian_operator(
    array: NDarray64, dx: float = 0.01, dy: float = 0.01
) -> NDarray64:
    """Computes laplacian operator on the array

    Returns:
        discrete laplacian operator as NDarray64
    """
    return (
        -2.0 * array[1:-1, 1:-1] + array[1:-1, :-2] + array[1:-1, 2:]
    ) / dx / dx + (
        -2.0 * array[1:-1, 1:-1] + array[:-2, 1:-1] + array[2:, 1:-1]
    ) / dy / dy


def compute_residual(
    residual: NDarray64, array: NDarray64, rhs: NDarray64
) -> None:
    """Performs computation of residual `r = b - Ax`

    Parameters:
        residual: residual array holder as NDarray64
        rhs: discrete array holding rhs values as NDArray64
    """
    residual[1:-1, 1:-1] = rhs[1:-1, 1:-1] - laplacian_operator(array)


def compute_laplacian_on_array(
    operator_holder: NDarray64, array: NDarray64
) -> None:
    """Performs laplacian computation on array in the coordinates
    of interest

    Parameters:
        operator_holder: array holding the laplacian operation
        array: input array to apply the laplacian to, as NDarray64
    """
    operator_holder[1:-1, 1:-1] = laplacian_operator(array)


def SteppestDescent(
    array: NDarray64,
    rhs: NDarray64,
    max_iter: int,
    relative_tol: float = 1e-6,
) -> tuple[NDarray64, int, list[np.float64]]:
    """Implements Gradient Descent algorithm, starting with initial solution
    array, right hand side (rhs) and parameters for convergence.

    Parameters:
        array: initial guess for solution array
        rhs: rhs of the discrete problem Ax = b
        max_iter: maximum number of iterations
        relative_tol: tolerance criteria for convergence

    Returns:
        solution: array as NDarray64
        i: iteration count for convergence
        history: history of relative errors
    """
    solution: NDarray64 = array.copy()
    residual: NDarray64 = np.zeros_like(solution)
    operator_on_residual: NDarray64 = np.zeros_like(solution)

    history: list[np.float64] = []
    diff: np.float64 = np.float64(1e8)
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
    array: NDarray64,
    rhs: NDarray64,
    max_iter: int,
    relative_tol: float = 1e-6,
) -> tuple[NDarray64, int, list[np.float64]]:
    """Implements Conjugate Gradient algorithm, starting with initial solution
    array, right hand side (rhs) and parameters for convergence.

    Parameters:
        array: initial guess for solution array
        rhs: rhs of the discrete problem Ax = b
        max_iter: maximum number of iterations
        relative_tol: tolerance criteria for convergence

    Returns:
        solution: array as NDarray64
        i: iteration count for convergence
        history: history of relative errors
    """
    solution: NDarray64 = array.copy()
    residual: NDarray64 = np.zeros_like(solution)
    operator_dot_direction: NDarray64 = np.zeros_like(solution)
    compute_residual(residual, solution, rhs)  # initial residual
    direction: NDarray64 = residual.copy()

    history: list[np.float64] = []
    diff: np.float64 = np.float64(1e8)
    i: int = 0
    while diff > relative_tol and i < max_iter:
        solution_k: NDarray64 = solution.copy()
        residual_k: NDarray64 = residual.copy()
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
    solution: NDarray64, reference: NDarray64
) -> np.float64:
    """Computes absolute error difference between two solutions"""
    return np.sqrt(np.sum((solution - reference) ** 2))


def compute_norm_to_reference(
    solution: NDarray64,
    reference: NDarray64,
    atol: float = 1e-12,
) -> np.float64:
    """Computes relative error between solution array and reference"""
    l2_difference = np.sqrt(np.sum((solution - reference) ** 2))
    l2_reference = np.sqrt(np.sum(reference**2))

    if l2_difference < atol or l2_reference < atol:
        return l2_difference
    else:
        return l2_difference / l2_reference


def create_mesh_for_analytical_solution(
    dx: float, dy: float, Lx: float, Ly: float
) -> tuple[NDarray64, NDarray64, NDarray64]:
    """Creates mesh for computation, solution array as NDarray64.
    Mesh is defined by the set of 2D coordinates X, Y

    Returns:
        X: x-coordinates as NDarray64
        Y: y-coordinates as NDarray64
        solution: NDarray64 solution array
    """
    x = np.arange(0.0, Lx, step=dx)
    y = np.arange(0.0, Ly, step=dy)
    X, Y = np.meshgrid(x, y)
    solution = np.zeros_like(X)
    add_boundary_conditions(solution, X, Y, Lx, Ly)

    return X, Y, solution


def create_rhs_for_analytical_solution(
    X: NDarray64, Y: NDarray64, Lx: float, Ly: float
) -> NDarray64:
    """Computes analytical solution as NDarray64 from
    coordinates X and Y"""
    return np.array(
        -(1 / Lx / Lx + 1 / Ly / Ly)
        * np.pi**2
        * np.sin(np.pi * X / Lx)
        * np.cos(np.pi * Y / Ly),
        dtype=np.float64,
    )


def test_compute_analytical_solution(
    dx: float = 0.01,
    dy: float = 0.01,
    Lx: float = 1.0,
    Ly: float = 1.0,
    path_to_test: pathlib.Path = pathlib.Path("tests"),
) -> None:
    x = np.arange(0.0, Lx, step=dx, dtype=np.float64)
    y = np.arange(0.0, Ly, step=dy, dtype=np.float64)
    X, Y = np.meshgrid(x, y)
    analytical_solution = compute_analytical_solution(X, Y, Lx, Ly)
    path_to_test.mkdir(parents=True, exist_ok=True)
    path_to_png = path_to_test.joinpath(
        f"analytical_solution_dx{dx}_dy{dy}.png"
    )
    save_array_to_png(X, Y, analytical_solution, path_to_png)


def test_conjugate_gradient(
    dx: float = 0.01,
    dy: float = 0.01,
    path_to_test: pathlib.Path = pathlib.Path("tests"),
) -> None:
    """Test Conjugate Gradient algorithm from known analytical solution"""
    Lx, Ly = 1.0, 1.0
    X, Y, solution = create_mesh_for_analytical_solution(dx, dy, Lx, Ly)
    rhs = create_rhs_for_analytical_solution(X, Y, Lx, Ly)
    max_iter, rtol = 1000, 1e-8
    solution, iterations, history = ConjugateGradientDescent(
        solution, rhs, max_iter, rtol
    )
    analytical_solution = compute_analytical_solution(X, Y, Lx, Ly)
    error_to_reference = compute_absolute_error_difference(
        solution, analytical_solution
    )
    path_to_test.mkdir(parents=True, exist_ok=True)
    path_to_png = path_to_test.joinpath(
        f"test_conjugate_gradient_dx{dx}_dy{dy}.png"
    )
    save_array_to_png(X, Y, solution, path_to_png)
    print(
        f"CG converged in {iterations}\n"
        f"with relative tolerance {history[-1]}\n"
        f"Absolute error to reference {error_to_reference}\n\n"
    )


def test_gradient_descent(
    dx: float = 0.01,
    dy: float = 0.01,
    path_to_test: pathlib.Path = pathlib.Path("tests"),
) -> None:
    """Test Gradient Descent algorithm from known analytical solution"""
    Lx, Ly = 1.0, 1.0
    X, Y, solution = create_mesh_for_analytical_solution(dx, dy, Lx, Ly)
    rhs = create_rhs_for_analytical_solution(X, Y, Lx, Ly)
    max_iter, rtol = 20000, 1e-8

    solution, iterations, history = SteppestDescent(
        solution, rhs, max_iter, rtol
    )
    analytical_solution = compute_analytical_solution(X, Y, Lx, Ly)
    error_to_reference = compute_absolute_error_difference(
        solution, analytical_solution
    )
    path_to_test.mkdir(parents=True, exist_ok=True)
    path_to_png: pathlib.Path = path_to_test.joinpath(
        f"test_steppest_descent_dx{dx}_dy{dy}.png"
    )
    save_array_to_png(X, Y, solution, path_to_png)
    print(
        f"Gradient Descent converged in {iterations}\n"
        f"with relative tolerance {history[-1]}\n"
        f"Absolute error to reference {error_to_reference}\n\n"
    )


def create_specific_rhs(
    X: NDarray64, Y: NDarray64, Lx: float, Ly: float
) -> NDarray64:
    """Creates complex sinusoidal right-hand side"""
    rhs_array = np.array(
        np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
        + 2 * np.sin(2 * np.pi * X / Lx) * np.sin(2 * np.pi * Y / Ly)
        + 3 * np.sin(3 * np.pi * X / Lx) * np.sin(3 * np.pi * Y / Ly),
        dtype=np.float64,
    )
    return rhs_array


def create_mesh_from_parameters(
    dx: float, dy: float, Lx: float, Ly: float
) -> tuple[NDarray64, NDarray64, NDarray64]:
    """Creates mesh and solution array from parameters"""
    x = np.arange(0.0, Lx, step=dx)
    y = np.arange(0.0, Ly, step=dy)
    X, Y = np.meshgrid(x, y)
    solution = np.zeros_like(X, dtype=np.float64)

    return X, Y, solution


def test_conjugate_gradient_specific_rhs(
    M: int,
    N: int,
    Lx: float | int = float(2),
    Ly: float | int = float(1),
    path_to_file: str | pathlib.Path = "results",
) -> None:
    """Computes test cases for using the conjugate gradient algorithm.
    Test is defined using parameters M, N, Lx, Ly and path_to_file.

    Parameters:
        M: discretization number of x-direction
        N: discretization number of y-direction
        Lx: length x-direction
        Ly: length y-direction
        path_to_file: path to store solutions
    """
    assert isinstance(M, int) and isinstance(N, int)
    assert M > 0 and N > 0
    assert isinstance(Lx, (int, float)) and isinstance(Ly, (int, float))
    assert Lx > 0 and Ly > 0
    dx: float = Lx / (M - 1)
    dy: float = Ly / (N - 1)
    max_iter, rtol = 5000, 1e-8

    X, Y, solution = create_mesh_from_parameters(dx, dy, Lx, Ly)
    rhs_array = create_specific_rhs(X, Y, Lx, Ly)
    solution, iterations, history = ConjugateGradientDescent(
        solution, rhs_array, max_iter, rtol
    )
    print(
        f"CG converged in {iterations}\n"
        f"with relative tolerance {history[-1]}\n"
    )
    save_solution(solution, X, Y, M, N, path_to_file)


def save_solution(
    solution_array: NDarray64,
    X: NDarray64,
    Y: NDarray64,
    M: float,
    N: float,
    path_to_file: str | pathlib.Path,
) -> None:
    """Saves solution field from specified path, checks if path exists,
    otherwise it creates a new one."""
    if isinstance(path_to_file, str):
        pathlib.Path(path_to_file).mkdir(parents=True, exist_ok=True)
        path_to_figure = pathlib.Path(path_to_file).joinpath(
            f"solution_M{M}_N{N}.png"
        )
        path_to_csv = pathlib.Path(path_to_file).joinpath(
            f"solution_M{M}_N{N}.csv"
        )
    elif isinstance(path_to_file, pathlib.Path):
        path_to_figure = path_to_file.joinpath(f"solution_M{M}_N{N}.png")
        path_to_csv = path_to_file.joinpath(f"solution_M{M}_N{N}.csv")
    else:
        raise TypeError("path_to_file has wrong type")

    np.savetxt(path_to_csv.as_posix(), solution_array, delimiter=",")
    save_array_to_png(X, Y, solution_array, path_to_figure)


def save_array_to_png(
    X: NDarray64, Y: NDarray64, solution: NDarray64, figure_path: pathlib.Path
) -> None:
    """Saves solution array as png file using 3D matplotlib"""
    fig = pyplot.figure(figsize=(8, 6))
    axes = mplot3d.Axes3D(fig)
    axes.set_xlabel("X coordinates")
    axes.set_ylabel("Y coordinates")
    axes.set_zlabel("Solution")
    axes.plot_surface(X, Y, solution, cmap=cm.viridis)
    axes.view_init(elev=32.0, azim=45.0)
    pyplot.savefig(figure_path.as_posix(), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    test_compute_analytical_solution()
    test_gradient_descent()
    test_conjugate_gradient()
    test_conjugate_gradient_specific_rhs(32, 16)
    test_conjugate_gradient_specific_rhs(256, 128)
    test_conjugate_gradient_specific_rhs(512, 256)
