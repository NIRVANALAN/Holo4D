"""Regressions for upstream timm/CoWTracker attention-constructor incompatibility."""
import types
from unittest.mock import Mock

import pytest
import torch

from holo4d.cowtracker_compat import configure_cowtracker_attention


class DummyVideoTransformer:
    def __init__(self, width=64, *, use_flash_attention3=True):
        self.width = width
        self.custom_fa3 = use_flash_attention3


def test_native_timm_attention_is_selected_at_construction(monkeypatch, tmp_path):
    # Leave the external package untouched; exercise the same constructor
    # call used by CoWTracker, including caller-supplied True and repeat setup.
    cls = type("VideoTransformer", (DummyVideoTransformer,), {})
    module = types.SimpleNamespace(VisionTransformerVideo=cls)
    monkeypatch.setattr("holo4d.cowtracker_compat.importlib.import_module", Mock(return_value=module))
    monkeypatch.setenv("COWTRACKER_DISABLE_FUSED_ATTN", "1")
    monkeypatch.setenv("COWTRACKER_SDPA_BACKEND", "flash")
    toggles = {}
    for name in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp", "enable_math_sdp"):
        toggles[name] = Mock()
        monkeypatch.setattr(torch.backends.cuda, name, toggles[name])
    configure_cowtracker_attention(tmp_path)
    configure_cowtracker_attention(tmp_path)
    value = cls(128, use_flash_attention3=True)
    assert value.width == 128
    assert value.custom_fa3 is False
    toggles["enable_flash_sdp"].assert_called_with(True)
    toggles["enable_math_sdp"].assert_called_with(False)


def test_invalid_attention_backend_fails_before_import(monkeypatch, tmp_path):
    monkeypatch.setenv("COWTRACKER_DISABLE_FUSED_ATTN", "1")
    monkeypatch.setenv("COWTRACKER_SDPA_BACKEND", "unknown")
    with pytest.raises(ValueError, match="Unknown COWTRACKER"):
        configure_cowtracker_attention(tmp_path)


def test_default_keeps_math_fallback_and_accepts_constructor_without_fa3(monkeypatch, tmp_path):
    class NoFA3Flag:
        def __init__(self, width=64):
            self.width = width
    monkeypatch.setattr('holo4d.cowtracker_compat.importlib.import_module',
                        Mock(return_value=types.SimpleNamespace(VisionTransformerVideo=NoFA3Flag)))
    monkeypatch.delenv('COWTRACKER_SDPA_BACKEND', raising=False)
    monkeypatch.setenv('COWTRACKER_DISABLE_FUSED_ATTN', '1')
    math, flash = Mock(), Mock()
    monkeypatch.setattr(torch.backends.cuda, 'enable_math_sdp', math)
    monkeypatch.setattr(torch.backends.cuda, 'enable_flash_sdp', flash)
    configure_cowtracker_attention(tmp_path)
    assert NoFA3Flag(128).width == 128
    flash.assert_called_with(True)
    math.assert_called_with(True)
