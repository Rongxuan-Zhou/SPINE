# Repository Guidelines

## Project Structure & Module Organization
Organize Python libraries in `spine/` with subpackages such as `perception/`, `planning/`, `control/`, and `logging/` to keep data flow explicit. ROS 2 nodes live in `ros2_ws/src/` (e.g., `spine_bringup/launch`), while simulation assets (URDFs, MJCF, meshes) reside in `assets/`. Keep reproducible configs in `configs/` (YAML or Hydra), experiment automation under `scripts/`, and notebooks in `notebooks/`. Mirror every module with a peer test (e.g., `spine/control/cbf_layer.py` → `tests/control/test_cbf_layer.py`).

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate`: isolate dependencies before editing.
- `pip install -e .[dev]`: installs the SPINE Python package plus lint/test extras.
- `colcon build --symlink-install`: builds ROS 2 nodes in `ros2_ws` with fast iteration.
- `make lint`: runs `black`, `ruff`, `mypy spine tests`, and `clang-format` over C++ nodes.
- `pytest tests -m "not slow"`: executes deterministic unit tests; drop the marker for full coverage.
- `ros2 launch spine_bringup sim.launch.py world:=assets/scenes/contact.sdf`: spins up the Mujoco/Isaac bridge for local validation.

## Coding Style & Naming Conventions
Use 4-space indentation, explicit type hints, and descriptive names (`compute_contact_cbf_gain`). Document coordinate frames, units, timing budgets, and safety limits in docstrings. Python adheres to `black` formatting; ROS 2 C++ targets `ament_clang_format` and `ament_cpplint`. Class names are `PascalCase`, functions `snake_case`, ROS packages `spine_<role>`, and YAML configs follow `domain_robot_variant.yaml`.

## Testing Guidelines
Author tests with `pytest` plus `pytest-asyncio` for ROS interfaces, placing fixtures in `tests/conftest.py`. GPU- or hardware-only cases must be marked `@pytest.mark.gpu` or `@pytest.mark.hardware` and skipped automatically when unavailable. Maintain ≥85 % coverage for `spine/physics` and guard real-robot actions with dry-run simulations before integration. Use deterministic seeds, assert tolerances via `numpy.isclose`, and capture log files under `artifacts/<timestamp>` for reproducibility.

## Commit & Pull Request Guidelines
Adopt an imperative short style: `area: action` (e.g., `control: enforce joint torque clamps`). Each PR should reference its issue, describe perception/planning/control changes, attach experiment IDs, and include screenshots or plots for policy metrics. Confirm `make lint`, `pytest`, and any `ros2 launch` smoke tests in the PR checklist, and keep commits narrowly scoped for easier bisection.

## Simulation & Safety Notes
Keep separate launch files for simulation (`sim.launch.py`) and hardware (`robot.launch.py`) with explicit emergency-stop topics and torque limits. Never push configs with live robot credentials; load them from `.env.local` via `python-dotenv`. When adding new planners or controllers, state assumptions about contact models and sampling rates directly in comments so downstream agents can evaluate risk quickly.
