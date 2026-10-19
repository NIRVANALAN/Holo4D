"""Keep incompatible upstream scorer dependencies out of inference extras."""
from pathlib import Path
import tomllib


def test_vbench_numpy1_environment_is_isolated():
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root/'pyproject.toml').read_text())
    assert 'numpy>=2.0,<2.8' in config['project']['dependencies']
    assert 'evo==1.37.0' in config['project']['optional-dependencies']['eval']
    for dependencies in config['project']['optional-dependencies'].values():
        assert not any(item.startswith('vbench') for item in dependencies)
    scorer = config['dependency-groups']['vbench']
    assert 'vbench==0.1.5' in scorer
    assert 'numpy>=1.26,<2' in scorer
    instructions = (root/'eval/README.md').read_text()
    assert 'pip install --group vbench' in instructions
    assert 'No editable Holo4D installation' in instructions
