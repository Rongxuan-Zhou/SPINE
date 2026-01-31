import subprocess
from pathlib import Path


def test_cli_runs_with_empty_sources(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        'output_dir: "{out}"\nmax_trajectories: 1\n'.format(out=str(tmp_path / "out")),
        encoding="utf-8",
    )

    result = subprocess.run(
        ["python", "scripts/run_kinematic_generator.py", "--config", str(cfg_path)],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "未配置任何数据源" in result.stderr or "未配置任何数据源" in result.stdout
