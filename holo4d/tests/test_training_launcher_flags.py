"""Training dense rendering is opt-in; inference's negative flag is invalid here."""
import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('launcher',['train_384p.sh','train_i2v_384p.sh'])
def test_training_launcher_uses_supported_default_off_render_branch(launcher):
    source = (ROOT/'train'/launcher).read_text()
    assert '--no_dense_branch' not in source
    assert '--use_dense_branch' not in source
    tree = ast.parse((ROOT/'train/train_track4d360.py').read_text())
    options = [node for node in ast.walk(tree)
               if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
               and node.func.attr == 'add_argument' and node.args
               and isinstance(node.args[0],ast.Constant)
               and node.args[0].value == '--use_dense_branch']
    assert len(options) == 1
    values = {kw.arg:ast.literal_eval(kw.value) for kw in options[0].keywords}
    assert values['action'] == 'store_true'
    assert values.get('default',False) is False
