import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin


def plot_validation():
    q_ref = np.load("data/fr3_q_ref.npy")
    q_opt = np.load("data/fr3_opt_result.npy")
    f_opt = np.load("data/fr3_opt_forces.npy")

    # Align to the same sliced window used in casadi_cito_mvp (accident zone)
    start_idx = 200
    end_idx = 300
    if len(q_ref) > end_idx:
        q_ref = q_ref[start_idx:end_idx]
    # q_opt/f_opt already correspond to the sliced window length T;
    # truncate if longer than ref window
    target_len = len(q_ref)
    if len(q_opt) > target_len:
        q_opt = q_opt[:target_len]
    if len(f_opt) > target_len:
        f_opt = f_opt[:target_len]

    model = pin.buildModelFromUrdf("models/fr3.urdf")
    data = model.createData()
    fid = model.getFrameId("fr3_link8")

    z_ref, z_opt = [], []
    for t in range(len(q_ref)):
        pin.framesForwardKinematics(model, data, q_ref[t])
        z_ref.append(data.oMf[fid].translation[2])
    for t in range(len(q_opt)):
        pin.framesForwardKinematics(model, data, q_opt[t])
        z_opt.append(data.oMf[fid].translation[2])

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("EE Z Height (m)", color="tab:blue")
    ax1.plot(z_ref, "--", color="tab:blue", label="Original (Ref)")
    ax1.plot(z_opt, "-", color="tab:blue", label="Optimized (CITO)")
    ax1.axhline(0.0, color="black", linestyle=":", label="Table Surface")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Contact Force (N)", color="tab:red")
    ax2.plot(f_opt, color="tab:red", alpha=0.6, label="Contact Force")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.legend(loc="upper right")

    plt.title("SPINE Physical Inpainting: Z Height vs Contact Force")
    plt.grid(True, alpha=0.3)
    plt.savefig("data/spine_validation.png")
    print("✅ Plot saved to data/spine_validation.png")


if __name__ == "__main__":
    plot_validation()
