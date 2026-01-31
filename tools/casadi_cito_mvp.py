"""
CasADi + Pinocchio CITO MVP (9DOF, finite-difference dynamics, plane contact).

Usage (conda env with pinocchio.casadi):
  conda run -n spine_opt python tools/generate_dummy_ref.py
  conda run -n spine_opt python tools/casadi_cito_mvp.py \
    --urdf models/fr3.urdf \
    --q-ref data/fr3_q_ref.npy \
    --time-steps 20 \
    --dt 0.05 \
    --contact-frame fr3_link8 \
    --table-height 0.0 \
    --mu 0.5 \
    --comp-penalty 1e-3

Outputs: data/fr3_opt_result.npy (q), data/fr3_opt_forces.npy (contact force)
"""

import argparse
import numpy as np
import casadi as ca
import pinocchio as pin
from pinocchio import casadi as cpin


def load_robot_model(urdf_path):
    """Load double model and convert to casadi model; build rnea function."""
    model_d = pin.buildModelFromUrdf(urdf_path)
    model = cpin.Model(model_d)
    data = model.createData()

    q_sym = ca.SX.sym("q", model.nq)
    v_sym = ca.SX.sym("v", model.nv)
    a_sym = ca.SX.sym("a", model.nv)
    cpin.rnea(model, data, q_sym, v_sym, a_sym)
    rnea_func = ca.Function("rnea", [q_sym, v_sym, a_sym], [data.tau])
    return model, data, rnea_func


def get_fk_func(model, data, frame_name):
    q_sym = ca.SX.sym("q", model.nq)
    cpin.framesForwardKinematics(model, data, q_sym)
    if not model.existFrame(frame_name):
        raise ValueError(f"Frame {frame_name} not found.")
    fid = model.getFrameId(frame_name)
    pos_sym = data.oMf[fid].translation
    return ca.Function("fk_pos", [q_sym], [pos_sym])


def get_jac_z_func(model, data, frame_name):
    """Build casadi function to compute d(z)/dq for the contact frame."""
    q_sym = ca.SX.sym("q", model.nq)
    cpin.framesForwardKinematics(model, data, q_sym)
    if not model.existFrame(frame_name):
        raise ValueError(f"Frame {frame_name} not found.")
    fid = model.getFrameId(frame_name)
    z_height = data.oMf[fid].translation[2]
    J_z_sym = ca.jacobian(z_height, q_sym)  # (1, nq)
    return ca.Function("jac_z", [q_sym], [J_z_sym])


def run_cito(args):
    # Load full trajectory; auto-detect deepest penetration window if not specified.
    q_all = np.load(args.q_ref)
    if args.start_idx is None or args.end_idx is None:
        model_d_tmp = pin.buildModelFromUrdf(args.urdf)
        data_d_tmp = model_d_tmp.createData()
        fid_tmp = model_d_tmp.getFrameId(args.contact_frame)
        z_vals = []
        for q in q_all:
            pin.framesForwardKinematics(model_d_tmp, data_d_tmp, q)
            z_vals.append(data_d_tmp.oMf[fid_tmp].translation[2])
        min_z_idx = int(np.argmin(z_vals))
        half_win = args.window_size // 2
        start_idx = max(0, min_z_idx - half_win)
        end_idx = min(len(q_all), min_z_idx + half_win)
        print(f"🤖 Auto-detected collision at frame {min_z_idx}; window [{start_idx}, {end_idx}]")
    else:
        start_idx = args.start_idx
        end_idx = args.end_idx
    if end_idx > len(q_all) or start_idx >= len(q_all):
        # fallback to head window if data is short or indexes invalid
        start_idx = 0
        end_idx = min(len(q_all), args.time_steps or len(q_all))
    q_ref = q_all[start_idx:end_idx]
    T, nq_data = q_ref.shape
    print(f"✂️ Slicing trajectory: frames {start_idx}–{end_idx} (T={T})")

    model, data, rnea_func = load_robot_model(args.urdf)
    fk_func = get_fk_func(model, data, args.contact_frame)
    jz_func = get_jac_z_func(model, data, args.contact_frame)

    nq = model.nq
    nv = model.nv
    if nq_data != nq:
        raise ValueError(f"q_ref dim {nq_data} != model nq {nq}")

    opti = ca.Opti()
    Q = opti.variable(nq, T)    # positions
    F = opti.variable(1, T)     # normal force

    Q_ref = opti.parameter(nq, T)
    opti.set_value(Q_ref, q_ref.T)

    mu = args.mu
    eps = args.comp_penalty
    # 放宽跟踪/互补以提高可行性
    w_tau = 0.1
    w_f = 0.01
    w_smooth = 1e-2
    w_u_arm = 1e-2
    w_u_gripper = 1e3
    weights_tau = np.array([w_u_arm] * min(7, nv) + [w_u_gripper] * max(0, nv - 7))

    cost = 0
    # Softly anchor the first frame (allow escape from penetration)
    cost += 1000.0 * ca.sumsqr(Q[:, 0] - q_ref[0, :])

    for t in range(T):
        q_t = Q[:, t]
        pos_t = fk_func(q_t)
        phi = pos_t[2] - args.table_height
        f_t = F[:, t]

        # Complementarity (relaxed but without penetration)
        opti.subject_to(f_t >= 0)
        # 不允许穿模；让接触在桌面或以上
        opti.subject_to(phi >= 0.0)
        opti.subject_to(f_t * phi <= eps)

        # Tracking and force regularization
        cost += 10.0 * ca.sumsqr(q_t - Q_ref[:, t])
        cost += w_f * ca.sumsqr(f_t)
        if t > 0:
            cost += w_smooth * ca.sumsqr(Q[:, t] - Q[:, t - 1])

        # Dynamics cost (finite-diff) for t < T-2
        if t < T - 2:
            q_next = Q[:, t + 1]
            q_next2 = Q[:, t + 2]
            v_t = (q_next - q_t) / args.dt
            v_next = (q_next2 - q_next) / args.dt
            a_t = (v_next - v_t) / args.dt

            tau_dyn = rnea_func(q_t, v_t, a_t)
            J_z = jz_func(q_t)  # (1, nq)
            tau_contact = J_z.T * f_t  # (nq,1)
            tau_motor = tau_dyn - tau_contact
            cost += ca.mtimes([tau_motor.T, ca.diag(weights_tau), tau_motor])

    opti.minimize(cost)

    p_opts = {"expand": True, "print_time": False}
    s_opts = {"max_iter": 300, "tol": 1e-4, "print_level": 5}
    opti.solver("ipopt", p_opts, s_opts)
    opti.set_initial(Q, q_ref.T)

    sol = opti.solve()
    q_opt = sol.value(Q).T
    f_opt = sol.value(F).T
    np.save("data/fr3_opt_result.npy", q_opt)
    np.save("data/fr3_opt_forces.npy", f_opt)
    print("✅ Optimization Success")
    print("Saved to data/fr3_opt_result.npy and data/fr3_opt_forces.npy")
    print("Contact force stats: min", f_opt.min(), "max", f_opt.max(), "mean", f_opt.mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, required=True)
    parser.add_argument("--q-ref", type=str, required=True)
    parser.add_argument("--time-steps", type=int, default=20)
    parser.add_argument("--start-idx", type=int, default=None, help="Slice start index (auto if None)")
    parser.add_argument("--end-idx", type=int, default=None, help="Slice end index (exclusive, auto if None)")
    parser.add_argument("--window-size", type=int, default=100, help="Auto window size when start/end not set")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--contact-frame", type=str, default="fr3_link8")
    parser.add_argument("--table-height", type=float, default=0.0)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--comp-penalty", type=float, default=1e-3, help="Relaxation epsilon")
    args = parser.parse_args()
    run_cito(args)
