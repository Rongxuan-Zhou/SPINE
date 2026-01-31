import h5py
import numpy as np
import robosuite as suite
import robosuite.macros as macros
import math
import argparse

macros.MUJOCO_GPU_RENDERING = False
macros.MUJOCO_EGL = False
macros.MUJOCO_GL = "glfw"


def rotmat_to_quat(mat):
    tr = np.trace(mat)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (mat[2, 1] - mat[1, 2]) / S
        qy = (mat[0, 2] - mat[2, 0]) / S
        qz = (mat[1, 0] - mat[0, 1]) / S
    else:
        if (mat[0, 0] > mat[1, 1]) and (mat[0, 0] > mat[2, 2]):
            S = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2
            qw = (mat[2, 1] - mat[1, 2]) / S
            qx = 0.25 * S
            qy = (mat[0, 1] + mat[1, 0]) / S
            qz = (mat[0, 2] + mat[2, 0]) / S
        elif mat[1, 1] > mat[2, 2]:
            S = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2
            qw = (mat[0, 2] - mat[2, 0]) / S
            qx = (mat[0, 1] + mat[1, 0]) / S
            qy = 0.25 * S
            qz = (mat[1, 2] + mat[2, 1]) / S
        else:
            S = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2
            qw = (mat[1, 0] - mat[0, 1]) / S
            qx = (mat[0, 2] + mat[2, 0]) / S
            qy = (mat[1, 2] + mat[2, 1]) / S
            qz = 0.25 * S
    return np.array([qw, qx, qy, qz])


def quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_axis_angle(q):
    qw, qx, qy, qz = q
    angle = 2 * math.acos(max(min(qw, 1.0), -1.0))
    s = math.sqrt(max(1 - qw * qw, 0.0))
    if s < 1e-8:
        axis = np.array([1, 0, 0])
    else:
        axis = np.array([qx, qy, qz]) / s
    return angle, axis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", type=str, default="data/mimicgen/pick_place.hdf5")
    ap.add_argument("--demo", type=str, default="demo_0")
    ap.add_argument("--site", type=str, default="gripper0_grip_site")
    ap.add_argument("--max-frames", type=int, default=200)
    args = ap.parse_args()

    with h5py.File(args.hdf5, "r") as h:
        obs = h["data"][args.demo]["obs"]
        jp = np.array(obs["robot0_joint_pos"])
        eef_quat_h5 = np.array(obs["robot0_eef_quat"])
    n = min(args.max_frames, jp.shape[0])

    env = suite.make(
        env_name="Lift", robots="Panda", has_renderer=False, use_camera_obs=False
    )
    env.reset()
    robot = env.robots[0]
    site_id = env.sim.model.site_name2id(args.site)

    angles = []
    first_offset = None
    for i in range(n):
        env.sim.data.qpos[robot._ref_joint_pos_indexes] = jp[i]
        env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
        env.sim.forward()
        mat = env.sim.data.site_xmat[site_id].reshape(3, 3)
        quat_sim = rotmat_to_quat(mat)
        inv_sim = quat_conjugate(quat_sim)
        q_offset = quat_multiply(eef_quat_h5[i], inv_sim)
        ang, axis = quat_to_axis_angle(q_offset / np.linalg.norm(q_offset))
        angles.append(ang)
        if i == 0:
            first_offset = q_offset
    print(
        f"Frames analyzed: {n}, mean ang error: {np.mean(angles):.3f} rad, max: {np.max(angles):.3f} rad"
    )
    print("First frame offset wxyz:", first_offset)


if __name__ == "__main__":
    main()
