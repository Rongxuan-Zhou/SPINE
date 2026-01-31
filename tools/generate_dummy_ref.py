import numpy as np
import os


def generate_dummy_9d():
    """
    FR3 7DOF arm + 2DOF gripper = 9 DOF trajectory for short-horizon tests.
    Gripper held fixed (e.g., closed) to avoid momentum hacking.
    """
    n_q_arm = 7
    T = 50

    # Arm initial pose (slightly lower for contact intent)
    q0_arm = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.5, 0.785])
    q_ref_arm = np.tile(q0_arm, (T, 1))

    # Small sinusoidal perturbations
    t = np.linspace(0, 4 * np.pi, T)
    q_ref_arm[:, 1] += 0.05 * np.sin(t)
    q_ref_arm[:, 3] += 0.05 * np.cos(t)

    # Gripper fixed (closed); change to 0.04 for open
    gripper_width = 0.0
    gripper_state = np.full((T, 2), gripper_width)

    q_ref_9d = np.hstack([q_ref_arm, gripper_state])  # shape (T, 9)

    os.makedirs("data", exist_ok=True)
    np.save("data/fr3_q_ref.npy", q_ref_9d)
    print(f"Generated 9-DOF trajectory data/fr3_q_ref.npy, shape={q_ref_9d.shape}")
    print(f"Last 2 dims (gripper) fixed at {gripper_width}")


if __name__ == "__main__":
    generate_dummy_9d()
