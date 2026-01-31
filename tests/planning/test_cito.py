import numpy as np

from spine.planning.cito import (
    CITOParameters,
    CITOPlanner,
    Trajectory,
    planar_contact,
    simple_point_mass_dynamics,
)


def test_cito_reduces_penetration() -> None:
    dt = 0.05
    params = CITOParameters(
        dt=dt,
        max_iters=10,
        contact_weight=50.0,
        track_weight=1.0,
        dynamics_weight=5.0,
        step_size=0.02,
        trust_region=0.05,
    )
    planner = CITOPlanner(params)
    dynamics = simple_point_mass_dynamics(dt)
    contact = planar_contact(boundary=0.0)

    # initial states: penetrating the boundary at x= -0.1
    states = np.array([[-0.1, 0.0], [-0.1, 0.0], [-0.05, 0.0]], dtype=float)
    controls = np.zeros((2, 1), dtype=float)
    target = np.array([[0.2, 0.0], [0.2, 0.0], [0.2, 0.0]], dtype=float)
    traj = Trajectory(states=states, controls=controls)

    result = planner.optimize(
        traj, dynamics_fn=dynamics, contact_fn=contact, target_states=target
    )

    # final positions should move towards +x and reduce penetration
    assert result.trajectory.states[:, 0].min() > -0.05
    assert result.costs[-1] < result.costs[0]
