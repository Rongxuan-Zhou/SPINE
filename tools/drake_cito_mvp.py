"""
Minimal Drake-based CITO prototype (quasi-static, single/multi contact) to
physically inpaint a short reference trajectory (e.g., MimicGen q_ref.npy).

Usage:
    python tools/drake_cito_mvp.py \
        --model models/panda_with_table.sdf \
        --q-ref data/q_ref.npy \
        --time-steps 20 \
        --table-height 0.0

Notes:
- This is a minimal, CPU-only, short-horizon prototype. It uses a quasi-static
  force balance (g ~= B u + sum J^T f) and a soft complementarity
  phi * fn <= eps. Upgrade to full collocation with AutoDiffXd for
  paper-quality results.
- Contact distance phi is simplified to "frame origin z minus table height".
  Replace `compute_phi` with a true signed-distance query (SceneGraph
  QueryObject) for complex geometries.
- Initial guess: q_ref is used to evaluate Jacobians and as tracking target.
  If your data penetrates the table, first run a geometric IK lift.
"""

import argparse
import numpy as np
from pydrake.all import (
    MathematicalProgram,
    MultibodyPlant,
    SceneGraph,
    Parser,
    SnoptSolver,
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
)


def build_diagram(model_path: str):
    """Build plant+scene_graph diagram for geometry queries."""
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    # Drake Python uses AddModelsFromUrl; use file:// URL.
    url = f"file://{model_path}"
    parser.AddModelsFromUrl(url)
    plant.Finalize()
    diagram = builder.Build()
    return plant, scene_graph, diagram


def compute_phi(plant: MultibodyPlant, scene_graph: SceneGraph, diagram, diagram_context, frame_name: str, q: np.ndarray, table_height: float, sd_threshold: float = 2.0) -> float:
    """Signed distance using SceneGraph QueryObject; fallback to plane z-table_height."""
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    scene_context = scene_graph.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, q)
    query_object = scene_graph.get_query_output_port().Eval(scene_context)
    p_W = plant.GetFrameByName(frame_name).CalcPose(plant_context, plant.world_frame()).translation()
    d_min = None
    try:
        distances = query_object.ComputeSignedDistanceToPoint(p_W, sd_threshold)
        if len(distances) > 0:
            d_min = min(d.distance for d in distances)
    except Exception:
        d_min = None
    if d_min is None:
        # fallback: plane at table_height
        return p_W[2] - table_height
    return d_min


def add_quasi_static_dynamics(prog, plant, q_ref, u, lam_n, lam_tx, lam_ty, contact_frames):
    """Approximate static force balance at each step: g(q) = B u + sum J^T f."""
    nq = plant.num_positions()
    nv = plant.num_velocities()
    nu = plant.num_actuators()
    assert q_ref.shape[1] == nq

    B = plant.MakeActuationMatrix()  # (nv x nu)
    T = q_ref.shape[0]
    for t in range(T):
        context = plant.CreateDefaultContext()
        plant.SetPositions(context, q_ref[t])
        tau_g = plant.CalcGravityGeneralizedForces(context)
        tau_contact = np.zeros(nv)
        for i, frame_name in enumerate(contact_frames):
            frame = plant.GetFrameByName(frame_name)
            J_spatial = plant.CalcJacobianSpatialVelocity(
                context,
                frame,
                np.zeros(3),
                plant.world_frame(),
                plant.world_frame(),
            )
            f_world = np.array([lam_tx[t, i], lam_ty[t, i], lam_n[t, i], 0.0, 0.0, 0.0])
            tau_contact += J_spatial.T @ f_world
        # Equality: tau_g = B u + tau_contact (B is nv x nu)
        prog.AddConstraint(tau_g == B @ u[t] + tau_contact)


def solve_cito(args):
    plant, scene_graph, diagram = build_diagram(args.model)
    diagram_context = diagram.CreateDefaultContext()
    nq, nv, nu = plant.num_positions(), plant.num_velocities(), plant.num_actuators()

    q_ref = np.load(args.q_ref)[: args.time_steps]
    T = q_ref.shape[0]

    contact_frames = args.contact_frames if args.contact_frames else ["panda_link7"]

    prog = MathematicalProgram()
    q = prog.NewContinuousVariables(T, nq, "q")
    v = prog.NewContinuousVariables(T, nv, "v")
    u = prog.NewContinuousVariables(T, nu, "u")
    lam_n = prog.NewContinuousVariables(T, len(contact_frames), "lam_n")
    lam_tx = prog.NewContinuousVariables(T, len(contact_frames), "lam_tx")
    lam_ty = prog.NewContinuousVariables(T, len(contact_frames), "lam_ty")

    # Hyper-parameters
    eps = 1e-4
    w_track, w_u, w_lam, w_smooth = 10.0, 1e-2, 1e-3, 1e-2
    mu = args.mu
    rho_comp = args.comp_penalty

    # Bounds
    prog.AddBoundingBoxConstraint(plant.GetPositionLowerLimits(), plant.GetPositionUpperLimits(), q.flatten())
    prog.AddBoundingBoxConstraint(plant.GetEffortLowerLimits(), plant.GetEffortUpperLimits(), u.flatten())

    for t in range(T):
        # Costs
        prog.AddQuadraticCost(w_track * np.sum((q[t] - q_ref[t]) ** 2))
        prog.AddQuadraticCost(w_u * np.sum(u[t] ** 2))
        prog.AddQuadraticCost(w_lam * (np.sum(lam_n[t] ** 2) + np.sum(lam_tx[t] ** 2) + np.sum(lam_ty[t] ** 2)))
        if t > 0:
            prog.AddQuadraticCost(w_smooth * np.sum((q[t] - q[t - 1]) ** 2))

        # Contact constraints (soft complementarity + friction cone)
        for i, frame_name in enumerate(contact_frames):
            phi = compute_phi(
                plant,
                scene_graph,
                diagram,
                diagram_context,
                frame_name,
                q_ref[t],  # using q_ref for distance approx; replace with decision-variable eval + AutoDiff for full fidelity
                args.table_height,
                args.sd_threshold,
            )
            prog.AddConstraint(lam_n[t, i] >= 0)
            prog.AddConstraint(phi >= 0)  # move to penalty if you allow small penetration
            prog.AddConstraint(lam_n[t, i] * phi <= eps)
            if rho_comp > 0:
                prog.AddQuadraticCost(rho_comp * (lam_n[t, i] * phi) ** 2)  # penalty version
            # Friction cone: ||ft|| <= mu * fn via Lorentz cone
            prog.AddLorentzConeConstraint([mu * lam_n[t, i], lam_tx[t, i], lam_ty[t, i]])

    add_quasi_static_dynamics(prog, plant, q_ref, u, lam_n, lam_tx, lam_ty, contact_frames)

    solver = SnoptSolver()
    result = solver.Solve(prog)
    if not result.is_success():
        raise RuntimeError("CITO solve failed")
    return result.GetSolution(q), result.GetSolution(u), result.GetSolution(lam_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to SDF/URDF containing robot+table.")
    parser.add_argument("--q-ref", required=True, help="Path to npy (T,nq) reference positions (e.g., MimicGen).")
    parser.add_argument("--time-steps", type=int, default=20, help="Number of steps to optimize.")
    parser.add_argument("--table-height", type=float, default=0.0, help="Table z in world frame.")
    parser.add_argument("--mu", type=float, default=0.5, help="Friction coefficient.")
    parser.add_argument(
        "--contact-frames",
        nargs="*",
        default=None,
        help="List of frame names to consider contacts (default: panda_link7).",
    )
    parser.add_argument(
        "--comp-penalty",
        type=float,
        default=0.0,
        help="Optional penalty rho on (phi*fn)^2 to soften complementarity.",
    )
    parser.add_argument(
        "--sd-threshold",
        type=float,
        default=2.0,
        help="Signed distance query threshold (m); if no result, fall back to plane distance.",
    )
    args = parser.parse_args()

    q_sol, u_sol, lam_sol = solve_cito(args)
    np.save("/tmp/cito_q.npy", q_sol)
    np.save("/tmp/cito_u.npy", u_sol)
    np.save("/tmp/cito_lambda.npy", lam_sol)
    print("CITO done. Saved to /tmp/cito_q.npy, /tmp/cito_u.npy, /tmp/cito_lambda.npy")
    print("lam max:", lam_sol.max(), "first q:", q_sol[0])


if __name__ == "__main__":
    main()
