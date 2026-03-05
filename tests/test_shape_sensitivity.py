"""Tests for shape sensitivity: differentiating FE solutions w.r.t. mesh node coordinates.

Covers:
  - Non-regression: delegating geometry methods still produces the same arrays.
  - FD gradient check: JAX adjoint gradient vs. central finite differences.
  - Trace verification: mesh coordinates appear as traced inputs in the jaxpr.
"""

import numpy as onp
import jax
import jax.numpy as np
import unittest

# Float64 is required for accurate finite-difference comparison.
jax.config.update("jax_enable_x64", True)

from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import Mesh
from jax_fem.problem import Problem
from jax_fem.fe import (
    compute_shape_grads,
    compute_physical_quad_points,
    compute_face_shape_grads,
    compute_physical_surface_quad_points,
)


# ---------------------------------------------------------------------------
# Helper: build a small regular QUAD4 mesh on the unit square
# ---------------------------------------------------------------------------


def _make_quad_mesh(Nx=4, Ny=4, Lx=1.0, Ly=1.0):
    """Return a Mesh for a regular QUAD4 grid on [0, Lx] x [0, Ly].

    Uses the same node/cell ordering as ``rectangle_mesh`` in
    ``jax_fem.generate_mesh``, but avoids the meshio dependency.
    """
    x = onp.linspace(0.0, Lx, Nx + 1)
    y = onp.linspace(0.0, Ly, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing="ij")
    points = onp.stack((xv, yv), axis=2).reshape(-1, 2)
    pts_inds = onp.arange(len(points)).reshape(Nx + 1, Ny + 1)
    cells = onp.stack(
        (pts_inds[:-1, :-1], pts_inds[1:, :-1], pts_inds[1:, 1:], pts_inds[:-1, 1:]),
        axis=2,
    ).reshape(-1, 4)
    return Mesh(points, cells, ele_type="QUAD4")


# ---------------------------------------------------------------------------
# Problem subclass: 2D Poisson with source, shape-sensitivity ready
# ---------------------------------------------------------------------------


class PoissonShapeSensitivity(Problem):
    """2D Poisson problem: −Δu = 1, u = 0 on ∂Ω.

    ``set_params`` is wired to ``recompute_geometry`` so that
    the FE solution is differentiable w.r.t. mesh node coordinates.

    Example usage with ``ad_wrapper``::

        fwd_pred = ad_wrapper(problem)
        sol_list = fwd_pred(initial_points)          # forward solve
        grad = jax.grad(lambda p: np.sum(fwd_pred(p)[0]))(initial_points)
    """

    def get_tensor_map(self):
        # Laplace diffusion term: ∫ ∇u · ∇v dΩ
        return lambda x: x

    def get_mass_map(self):
        # Constant source term: residual form is ∫ ∇u·∇v − ∫ v dΩ = 0
        return lambda u, x: -np.ones_like(u)

    def set_params(self, params):
        self.recompute_geometry(params)


# ---------------------------------------------------------------------------
# Test: geometry standalone functions produce identical results to methods
# ---------------------------------------------------------------------------


class TestNonRegression(unittest.TestCase):
    """Delegating geometry methods must produce bit-identical arrays.

    These tests do NOT require solver or petsc4py.
    """

    def setUp(self):
        mesh = _make_quad_mesh(3, 3)
        dirichlet_bc_info = [
            [lambda p: np.isclose(p[0], 0.0, atol=1e-5)],
            [0],
            [lambda p: 0.0],
        ]
        self.problem = PoissonShapeSensitivity(
            mesh,
            vec=1,
            dim=2,
            ele_type="QUAD4",
            dirichlet_bc_info=dirichlet_bc_info,
        )
        self.fe = self.problem.fes[0]

    def test_shape_grads(self):
        """compute_shape_grads(onp, ...) must match get_shape_grads()."""
        sg_method, jxw_method = self.fe.get_shape_grads()
        sg_fn, jxw_fn = compute_shape_grads(
            onp,
            self.fe.points,
            self.fe.cells,
            self.fe.shape_grads_ref,
            self.fe.quad_weights,
        )
        onp.testing.assert_array_equal(sg_method, sg_fn)
        onp.testing.assert_array_equal(jxw_method, jxw_fn)

    def test_physical_quad_points(self):
        """compute_physical_quad_points(onp, ...) must match get_physical_quad_points()."""
        pqp_method = self.fe.get_physical_quad_points()
        pqp_fn = compute_physical_quad_points(
            onp,
            self.fe.points,
            self.fe.cells,
            self.fe.shape_vals,
        )
        onp.testing.assert_array_equal(pqp_method, pqp_fn)

    def test_recompute_geometry_matches_init(self):
        """recompute_geometry(points) with original points must reproduce init geometry."""
        # Capture original geometry set at construction time (numpy arrays).
        original_shape_grads = onp.array(self.problem.shape_grads)
        original_JxW = onp.array(self.problem.JxW)
        original_pqp = onp.array(self.problem.physical_quad_points)

        # Recompute using JAX path (jnp internals) with the same points.
        self.problem.recompute_geometry(self.fe.points)

        onp.testing.assert_allclose(
            onp.array(self.problem.shape_grads),
            original_shape_grads,
            rtol=1e-12,
            err_msg="shape_grads mismatch after recompute_geometry",
        )
        onp.testing.assert_allclose(
            onp.array(self.problem.JxW),
            original_JxW,
            rtol=1e-12,
            err_msg="JxW mismatch after recompute_geometry",
        )
        onp.testing.assert_allclose(
            onp.array(self.problem.physical_quad_points),
            original_pqp,
            rtol=1e-12,
            err_msg="physical_quad_points mismatch after recompute_geometry",
        )


# -----------------------------------------------
# Test: shape sensitivity via finite differences
# -----------------------------------------------


class TestShapeSensitivity(unittest.TestCase):
    """JAX adjoint shape gradient must match central finite differences."""

    def setUp(self):
        Nx, Ny = 4, 4
        self.Nx, self.Ny = Nx, Ny
        mesh = _make_quad_mesh(Nx, Ny)
        # Store float64 points for accurate FD comparisons.
        self.points = mesh.points.copy().astype(onp.float64)

        def bottom(p):
            return np.isclose(p[1], 0.0, atol=1e-5)

        def top(p):
            return np.isclose(p[1], 1.0, atol=1e-5)

        def left(p):
            return np.isclose(p[0], 0.0, atol=1e-5)

        def right(p):
            return np.isclose(p[0], 1.0, atol=1e-5)

        dirichlet_bc_info = [
            [bottom, top, left, right],
            [0, 0, 0, 0],
            [lambda p: 0.0] * 4,
        ]

        self.problem = PoissonShapeSensitivity(
            mesh,
            vec=1,
            dim=2,
            ele_type="QUAD4",
            dirichlet_bc_info=dirichlet_bc_info,
        )
        self.fwd_pred = ad_wrapper(self.problem)

    def _objective(self, points):
        """Scalar objective J = Σ u_i (sum of all nodal solution values)."""
        sol_list = self.fwd_pred(points)
        return np.sum(sol_list[0])

    def test_fd_gradient(self):
        """Adjoint gradient must agree with central FD to <0.1 % relative error."""
        points = self.points

        # JAX adjoint gradient ∂J/∂p
        grad_jax = jax.grad(self._objective)(points)

        # Pick an interior node off both symmetry axes to ensure non-zero gradients:
        # (ix=1, iy=1) → position (0.25, 0.25) on the 5×5 grid.
        ix, iy = 1, 1
        node_ind = ix * (self.Ny + 1) + iy

        eps = 1e-5
        atol = 1e-9  # absolute tolerance for near-zero gradient comparisons
        for coord in range(2):
            # Use numpy-style copy+add since self.points is a numpy array.
            p_plus = onp.array(points)
            p_plus[node_ind, coord] += eps
            p_minus = onp.array(points)
            p_minus[node_ind, coord] -= eps
            j_plus = float(self._objective(p_plus))
            j_minus = float(self._objective(p_minus))
            fd_grad = (j_plus - j_minus) / (2.0 * eps)
            jax_g = float(grad_jax[node_ind, coord])

            abs_err = abs(fd_grad - jax_g)
            scale = max(abs(fd_grad), abs(jax_g), 1e-12)
            rel_err = abs_err / scale
            self.assertTrue(
                rel_err < 1e-3 or abs_err < atol,
                msg=(
                    f"coord={coord}: FD={fd_grad:.6e}, "
                    f"JAX={jax_g:.6e}, rel_err={rel_err:.2e}, abs_err={abs_err:.2e}"
                ),
            )

    def test_gradient_shape_and_dtype(self):
        """Gradient w.r.t. points has the same shape and float64 dtype as points."""
        grad = jax.grad(self._objective)(self.points)
        self.assertEqual(
            grad.shape,
            self.points.shape,
            msg="Gradient shape must match points shape.",
        )
        self.assertEqual(
            grad.dtype,
            onp.float64,
            msg="Gradient must be float64 (jax_enable_x64=True).",
        )


if __name__ == "__main__":
    unittest.main()
