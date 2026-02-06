"""Pure-Python approximation of Contact-Implicit Trajectory Optimization (CITO).

This module implements a lightweight VSCM-style contact cost and a successive
linearization loop to warm-start physics infill without relying on the C++
catkin/SNOPT pipeline.
"""

from .configs import CITOParameters
from .models import planar_contact, simple_point_mass_dynamics
from .solver import CITOPlanResult, CITOPlanner, Trajectory
from .projection import project_trajectory

__all__ = [
    "CITOParameters",
    "CITOPlanner",
    "CITOPlanResult",
    "Trajectory",
    "planar_contact",
    "simple_point_mass_dynamics",
    "project_trajectory",
]
