import pytest

from spine.perception.kinematics.configs import MimicGenConfig
from spine.perception.kinematics.dependencies import require_dependency
from spine.perception.kinematics.sources_mimicgen import MimicGenAdapter


def test_require_dependency_points_to_external_dir() -> None:
    with pytest.raises(ImportError) as exc:
        require_dependency(
            module_name="definitely_missing_adapter_dep",
            repo_name="MissingLib",
            repo_url="https://example.com/missing",
            install_subdir="missinglib",
        )
    message = str(exc.value)
    assert "external/missinglib" in message
    assert "pip install -e" in message


def test_mimicgen_adapter_prompts_for_dependency(tmp_path) -> None:
    adapter = MimicGenAdapter(MimicGenConfig(dataset_root=tmp_path))
    with pytest.raises((ImportError, FileNotFoundError)):
        list(adapter.generate())
