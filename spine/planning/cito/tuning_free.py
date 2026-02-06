"""Tuning-free CITO implementation (SCVX + VSCM-style complementarity).

This module provides a practical, fully-Python tuning-free CITO solver suitable
for large-scale offline trajectory inpainting. It uses:
  - MuJoCo for kinematics (contact distance + Jacobian)
  - Sequential convexification (SCVX) solved via CVXPY/OSQP
  - A penalty loop that anneals complementarity tolerance and slack weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

import cvxpy as cp
import mujoco


Array = np.ndarray


@dataclass
class TuningFreeCITOConfig:
    """Hyperparameters for tuning-free CITO (SCVX + penalty loop)."""

    dt: float = 0.05
    scvx_iters: int = 5
    penalty_loops: int = 6
    trust_region: float = 0.1
    trust_region_vel: float = 0.2
    trust_region_tau: float = 20.0
    track_weight: float = 10.0
    track_vel_weight: float = 1.0
    control_weight: float = 1e-3
    tau_weight: float = 0.0
    force_weight: float = 1e-3
    dynamics_weight: float = 1.0
    slack_weight: float = 1.0
    comp_slack_weight: float = 1.0
    epsilon_start: float = 1e-2
    epsilon_min: float = 1e-6
    epsilon_decay: float = 0.2
    slack_weight_scale: float = 10.0
    comp_weight_scale: float = 10.0
    fd_eps: float = 1e-4
    tol_penetration: float = 1e-4
    tol_comp: float = 1e-4
    solver: str = "OSQP"
    solver_verbose: bool = False
    solver_fallbacks: tuple[str, ...] = ("SCS", "CLARABEL")


@dataclass
class TuningFreeCITOResult:
    q_opt: Array
    u_opt: Array
    lambda_opt: Array
    tau_opt: Array
    v_opt: Array
    diagnostics: list[dict[str, float]]


class TuningFreeCITO:
    """SCVX-based tuning-free CITO using MuJoCo kinematics."""

    def __init__(
        self,
        model_xml: str,
        contact_site: Optional[str] = None,
        contact_sites: Optional[Iterable[str]] = None,
        contact_geoms: Optional[Iterable[str]] = None,
        contact_box_geom: Optional[str] = None,
        contact_box_body: Optional[str] = None,
        use_table_plane: bool = True,
        table_height: float = 0.0,
        joint_prefix: str = "robot0_joint",
        config: Optional[TuningFreeCITOConfig] = None,
    ) -> None:
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        self.config = config or TuningFreeCITOConfig()
        self.table_height = float(table_height)
        self.use_table_plane = bool(use_table_plane)

        # Determine contact points (sites / geoms)
        self.contact_points: list[tuple[str, int, str]] = []
        sites = list(contact_sites) if contact_sites is not None else []
        geoms = list(contact_geoms) if contact_geoms is not None else []
        if contact_site:
            sites = [contact_site] + [s for s in sites if s != contact_site]
        if not sites and not geoms:
            sites = [self._auto_contact_site()]
        for name in sites:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            self.contact_points.append(("site", sid, name))
        for name in geoms:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            self.contact_points.append(("geom", gid, name))

        self.contact_site = None
        for kind, _idx, name in self.contact_points:
            if kind == "site":
                self.contact_site = name
                break

        # Optional box surface (for peg/nut contact)
        self.box_geom_id: Optional[int] = None
        self.box_geom_name: Optional[str] = None
        if contact_box_geom or contact_box_body:
            if contact_box_geom:
                gid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, contact_box_geom
                )
            else:
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, contact_box_body
                )
                geom_ids = [
                    gid
                    for gid in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[gid]) == int(bid)
                ]
                if not geom_ids:
                    raise ValueError(f"No geoms found for body '{contact_box_body}'")
                gid = geom_ids[0]
            gtype = int(self.model.geom_type[gid])
            if gtype not in (
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
            ):
                raise ValueError(
                    "contact_box_geom/body must be box/cylinder geom"
                )
            self.box_geom_id = gid
            self.box_geom_name = (
                contact_box_geom if contact_box_geom else f"{contact_box_body}:{gid}"
            )

        # Determine joint indices / dof indices for the robot arm
        self.joint_ids = self._find_joint_ids(joint_prefix)
        self.joint_qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=int
        )
        self.joint_dof_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=int
        )

    def _auto_contact_site(self) -> str:
        # Preferred ordering of likely end-effector sites
        preferred = [
            "gripper0_ft_frame",
            "gripper0_grip_site",
            "robot0_grip_site",
        ]
        for name in preferred:
            try:
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
                return name
            except Exception:
                continue
        # Fallback: first site containing "gripper"
        for sid in range(self.model.nsite):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, sid)
            if name and "gripper" in name:
                return name
        # Last fallback: first site
        if self.model.nsite > 0:
            return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, 0)
        raise ValueError("No sites found in MuJoCo model to use for contact.")

    def _find_joint_ids(self, prefix: str) -> list[int]:
        joint_ids = []
        for jid in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            if name and name.startswith(prefix):
                joint_ids.append(jid)
        if not joint_ids:
            raise ValueError(f"No joints found with prefix '{prefix}' in model.")
        return joint_ids

    def _set_qpos(self, q: Array) -> None:
        if q.shape[0] != len(self.joint_qpos_adr):
            raise ValueError(
                f"q dim {q.shape[0]} != joint count {len(self.joint_qpos_adr)}"
            )
        self.data.qpos[self.joint_qpos_adr] = q
        self.data.qvel[:] = 0.0

    def _box_sdf_and_grad(self, p_world: Array) -> tuple[float, Array]:
        if self.box_geom_id is None:
            raise RuntimeError("box geom not configured")
        gid = self.box_geom_id
        center = self.data.geom_xpos[gid]
        xmat = self.data.geom_xmat[gid].reshape(3, 3)
        half = self.model.geom_size[gid].copy()
        if int(self.model.geom_type[gid]) == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # approximate cylinder as box in x-y
            half = np.array([half[0], half[0], half[1]], dtype=float)
        p_local = xmat.T @ (p_world - center)
        q = np.abs(p_local) - half
        outside = np.maximum(q, 0.0)
        outside_norm = float(np.linalg.norm(outside))
        if outside_norm > 0:
            grad_local = outside / outside_norm
            grad_local *= np.sign(p_local)
        else:
            idx = int(np.argmax(q))
            grad_local = np.zeros(3, dtype=float)
            grad_local[idx] = np.sign(p_local[idx]) if p_local[idx] != 0 else 1.0
        inside = float(np.minimum(np.max(q), 0.0))
        sd = outside_norm + inside
        grad_world = xmat @ grad_local
        return sd, grad_world

    def _contact_kinematics(self, q: Array) -> tuple[Array, Array]:
        """Return signed distances and Jacobians for all configured contacts."""
        self._set_qpos(q)
        mujoco.mj_forward(self.model, self.data)

        positions: list[Array] = []
        jacps: list[Array] = []
        for kind, idx, _name in self.contact_points:
            jacp = np.zeros((3, self.model.nv), dtype=float)
            jacr = np.zeros((3, self.model.nv), dtype=float)
            if kind == "site":
                pos = self.data.site_xpos[idx].copy()
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, idx)
            else:
                pos = self.data.geom_xpos[idx].copy()
                mujoco.mj_jacGeom(self.model, self.data, jacp, jacr, idx)
            positions.append(pos)
            jacps.append(jacp[:, self.joint_dof_adr].copy())

        phi_list: list[float] = []
        j_list: list[Array] = []

        for pos, jacp in zip(positions, jacps):
            if self.use_table_plane:
                phi_plane = float(pos[2] - self.table_height)
                j_plane = jacp[2].copy()
                phi_list.append(phi_plane)
                j_list.append(j_plane)
            if self.box_geom_id is not None:
                phi_box, grad_world = self._box_sdf_and_grad(pos)
                j_box = grad_world @ jacp
                phi_list.append(float(phi_box))
                j_list.append(j_box.copy())

        if not phi_list:
            raise RuntimeError("No contact constraints configured")
        phi = np.array(phi_list, dtype=float)
        j = np.stack(j_list, axis=0)
        return phi, j

    def _inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        self._set_qpos(q)
        self.data.qvel[:] = 0.0
        self.data.qvel[self.joint_dof_adr] = v
        self.data.qacc[:] = 0.0
        self.data.qacc[self.joint_dof_adr] = a
        mujoco.mj_inverse(self.model, self.data)
        tau = self.data.qfrc_inverse[self.joint_dof_adr].copy()
        return tau

    def _linearize_inverse_dynamics(
        self, q: Array, v: Array, a: Array, eps: float
    ) -> tuple[Array, Array, Array, Array]:
        """MuJoCo linearization of inverse dynamics.

        Returns tau0, Aq, Av, Aa where:
          tau(q,v,a) ≈ tau0 + Aq (q-q0) + Av (v-v0) + Aa (a-a0)
        """
        self._set_qpos(q)
        self.data.qvel[:] = 0.0
        self.data.qvel[self.joint_dof_adr] = v
        self.data.qacc[:] = 0.0
        self.data.qacc[self.joint_dof_adr] = a
        mujoco.mj_inverse(self.model, self.data)

        tau0_full = self.data.qfrc_inverse.copy()
        nv = self.model.nv
        dfdq = np.zeros((nv, nv), dtype=float)
        dfdv = np.zeros((nv, nv), dtype=float)
        dfda = np.zeros((nv, nv), dtype=float)
        mujoco.mjd_inverseFD(
            self.model,
            self.data,
            eps,
            0,
            dfdq,
            dfdv,
            dfda,
            None,
            None,
            None,
            None,
        )
        idx = self.joint_dof_adr
        tau0 = tau0_full[idx].copy()
        Aq = dfdq[np.ix_(idx, idx)].copy()
        Av = dfdv[np.ix_(idx, idx)].copy()
        Aa = dfda[np.ix_(idx, idx)].copy()
        return tau0, Aq, Av, Aa

    def _solve_problem(
        self,
        problem: cp.Problem,
        solver: str,
        fallbacks: Sequence[str],
        verbose: bool,
    ) -> str:
        solvers = [solver] + [s for s in fallbacks if s != solver]
        installed = set(cp.installed_solvers())
        last_error: Exception | None = None
        for candidate in solvers:
            if candidate not in installed:
                continue
            try:
                problem.solve(solver=candidate, verbose=verbose)
            except Exception as exc:
                last_error = exc
                continue
            if problem.status in ("optimal", "optimal_inaccurate"):
                return candidate
        if last_error is not None:
            raise RuntimeError(
                f"SCVX subproblem failed. Last solver error: {last_error}"
            )
        raise RuntimeError("SCVX subproblem failed to solve with available solvers.")

    def solve(self, q_ref: Array) -> TuningFreeCITOResult:
        """Run tuning-free CITO to inpaint a joint-space trajectory."""
        cfg = self.config
        T, nq = q_ref.shape
        if nq != len(self.joint_qpos_adr):
            raise ValueError(
                f"q_ref dim {nq} != expected joints {len(self.joint_qpos_adr)}"
            )

        # Initialize with reference
        q_k = q_ref.copy()
        v_ref = np.zeros_like(q_ref)
        v_ref[:-1] = (q_ref[1:] - q_ref[:-1]) / cfg.dt
        v_ref[-1] = v_ref[-2] if T > 1 else 0.0
        v_k = v_ref.copy()
        u_k = np.zeros((T - 1, nq), dtype=float)
        a_ref = np.zeros_like(q_ref)
        a_ref[:-1] = (v_ref[1:] - v_ref[:-1]) / cfg.dt
        a_ref[-1] = a_ref[-2] if T > 1 else 0.0
        phi0, _ = self._contact_kinematics(q_k[0])
        n_contacts = int(phi0.shape[0])
        lam_k = np.full((T, n_contacts), 0.1, dtype=float)
        tau_k = np.zeros((T, nq), dtype=float)
        for t in range(T):
            tau_k[t] = self._inverse_dynamics(q_ref[t], v_ref[t], a_ref[t])

        eps = cfg.epsilon_start
        slack_w = cfg.slack_weight
        comp_slack_w = cfg.comp_slack_weight
        diagnostics: list[dict[str, float]] = []

        for p_it in range(cfg.penalty_loops):
            for scvx_it in range(cfg.scvx_iters):
                # Linearize contact at current trajectory
                phi_k = np.zeros((T, n_contacts), dtype=float)
                J_k = np.zeros((T, n_contacts, nq), dtype=float)
                tau0_k = np.zeros((T - 1, nq), dtype=float)
                Aq_k = np.zeros((T - 1, nq, nq), dtype=float)
                Av_k = np.zeros((T - 1, nq, nq), dtype=float)
                Aa_k = np.zeros((T - 1, nq, nq), dtype=float)
                for t in range(T):
                    phi_k[t], J_k[t] = self._contact_kinematics(q_k[t])
                    if t < T - 1:
                        a_k = (v_k[t + 1] - v_k[t]) / cfg.dt
                        tau0, Aq, Av, Aa = self._linearize_inverse_dynamics(
                            q_k[t], v_k[t], a_k, cfg.fd_eps
                        )
                        tau0_k[t] = tau0
                        Aq_k[t] = Aq
                        Av_k[t] = Av
                        Aa_k[t] = Aa

                # Convex subproblem
                q = cp.Variable((T, nq))
                v = cp.Variable((T, nq))
                tau = cp.Variable((T, nq))
                u = cp.Variable((T - 1, nq))
                lam = cp.Variable((T, n_contacts))
                slack = cp.Variable((T, n_contacts))
                comp_slack = cp.Variable((T, n_contacts))

                constraints = []
                constraints.append(q[0] == q_ref[0])
                constraints.append(lam >= 0)
                constraints.append(slack >= 0)
                constraints.append(comp_slack >= 0)

                # Dynamics + trust region
                for t in range(T - 1):
                    constraints.append(q[t + 1] == q[t] + cfg.dt * v[t])
                    constraints.append(
                        cp.abs(q[t] - q_k[t]) <= cfg.trust_region
                    )
                    constraints.append(
                        cp.abs(v[t] - v_k[t]) <= cfg.trust_region_vel
                    )
                    constraints.append(
                        cp.abs(tau[t] - tau_k[t]) <= cfg.trust_region_tau
                    )
                constraints.append(cp.abs(q[T - 1] - q_k[T - 1]) <= cfg.trust_region)
                constraints.append(
                    cp.abs(v[T - 1] - v_k[T - 1]) <= cfg.trust_region_vel
                )
                constraints.append(
                    cp.abs(tau[T - 1] - tau_k[T - 1]) <= cfg.trust_region_tau
                )

                # Contact constraints (linearized)
                phi_lin = []
                comp_lin = []
                for t in range(T):
                    for c in range(n_contacts):
                        phi_t = phi_k[t, c] + J_k[t, c] @ (q[t] - q_k[t])
                        phi_lin.append(phi_t)
                        # non-penetration (with slack)
                        constraints.append(phi_t + slack[t, c] >= 0.0)

                        # linearized complementarity: lambda * phi <= eps
                        comp_t = (
                            lam_k[t, c] * phi_t
                            + phi_k[t, c] * (lam[t, c] - lam_k[t, c])
                            - lam_k[t, c] * phi_k[t, c]
                        )
                        comp_lin.append(comp_t)
                        constraints.append(comp_t <= eps + comp_slack[t, c])

                # Objective
                obj = 0
                obj += cfg.track_weight * cp.sum_squares(q - q_ref)
                obj += cfg.track_vel_weight * cp.sum_squares(v - v_ref)
                obj += cfg.control_weight * cp.sum_squares(u)
                obj += cfg.tau_weight * cp.sum_squares(tau)
                obj += cfg.force_weight * cp.sum_squares(lam)
                obj += slack_w * cp.sum_squares(slack)
                obj += comp_slack_w * cp.sum_squares(comp_slack)

                # Dynamics constraints (linearized inverse dynamics)
                for t in range(T - 1):
                    a_lin = (v[t + 1] - v[t]) / cfg.dt
                    a_k = (v_k[t + 1] - v_k[t]) / cfg.dt
                    tau_lin = (
                        tau0_k[t]
                        + Aq_k[t] @ (q[t] - q_k[t])
                        + Av_k[t] @ (v[t] - v_k[t])
                        + Aa_k[t] @ (a_lin - a_k)
                    )
                    tau_contact = 0
                    for c in range(n_contacts):
                        tau_contact += J_k[t, c].T * lam[t, c]
                    constraints.append(tau_lin - tau_contact - tau[t] == 0)

                problem = cp.Problem(cp.Minimize(obj), constraints)
                self._solve_problem(
                    problem,
                    solver=cfg.solver,
                    fallbacks=cfg.solver_fallbacks,
                    verbose=cfg.solver_verbose,
                )

                if q.value is None or lam.value is None or v.value is None or tau.value is None:
                    raise RuntimeError("SCVX subproblem failed to solve.")

                q_k = q.value
                v_k = v.value
                tau_k = tau.value
                u_k = u.value
                lam_k = lam.value

            # Evaluate residuals for penalty loop
            phi_eval = np.zeros((T, n_contacts), dtype=float)
            comp_eval = np.zeros((T, n_contacts), dtype=float)
            for t in range(T):
                phi_eval[t], _ = self._contact_kinematics(q_k[t])
                comp_eval[t] = lam_k[t] * phi_eval[t]

            max_pen = float(np.min(phi_eval))
            max_comp = float(np.max(np.abs(comp_eval)))
            diagnostics.append(
                {
                    "penalty_iter": float(p_it),
                    "epsilon": float(eps),
                    "max_penetration": float(max_pen),
                    "max_comp": float(max_comp),
                }
            )

            if (abs(max_pen) <= cfg.tol_penetration) and (
                max_comp <= cfg.tol_comp
            ):
                break

            # Penalty loop updates (tuning-free)
            eps = max(cfg.epsilon_min, eps * cfg.epsilon_decay)
            slack_w *= cfg.slack_weight_scale
            comp_slack_w *= cfg.comp_weight_scale

        return TuningFreeCITOResult(
            q_opt=q_k,
            u_opt=u_k,
            lambda_opt=lam_k,
            tau_opt=tau_k,
            v_opt=v_k,
            diagnostics=diagnostics,
        )


__all__ = ["TuningFreeCITOConfig", "TuningFreeCITO", "TuningFreeCITOResult"]
