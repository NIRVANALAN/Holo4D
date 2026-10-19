"""
Per-block track injection following ReCamMaster's pattern.

Two modes:

  - "additive" (default, original):
      track_encoder: Linear(track_dim, dit_dim), zero-init → starts as no-op
      injection: x = x + track_encoder(track_tokens)

  - "concat_project" (Koichi / Go-with-the-Track style):
      concat_proj: Linear(dit_dim + track_dim, dit_dim)
        - x-portion initialized to identity
        - track-portion initialized to zero
      injection: x = concat_proj(cat(x, track_tokens))
      At init this is identity on x (track contribution = 0), gradually
      learns to incorporate track signal. More expressive than additive.
"""

import os

import torch
import torch.nn as nn

from holo4d.utils.torch_profile import track4d_record_function


class PerBlockTrackInjector(nn.Module):
    """Per-block track encoders for injection at each DiT block.

    Args:
        num_blocks: Number of DiT blocks.
        track_dim: Dimension of track tokens.
        dit_dim: Dimension of DiT hidden states.
        mode: "additive" (original) or "concat_project" (Koichi-style).
    """

    def __init__(self, num_blocks: int, track_dim: int, dit_dim: int, mode: str = "additive"):
        super().__init__()
        self.num_blocks = num_blocks
        self.track_dim = track_dim
        self.dit_dim = dit_dim
        self.mode = mode

        if mode == "additive":
            self.encoders = nn.ModuleList([
                nn.Linear(track_dim, dit_dim) for _ in range(num_blocks)
            ])
            # Zero-init: at start, injection is a no-op (adds 0)
            for enc in self.encoders:
                nn.init.zeros_(enc.weight)
                nn.init.zeros_(enc.bias)

        elif mode == "concat_project":
            self.encoders = nn.ModuleList([
                nn.Linear(dit_dim + track_dim, dit_dim) for _ in range(num_blocks)
            ])
            # Identity-init on x-portion, zero-init on track-portion:
            # W = [W_x | W_track], b = 0
            # W_x = I → passes x through unchanged at init
            # W_track = 0 → track signal contributes nothing at init
            for enc in self.encoders:
                nn.init.zeros_(enc.weight)
                nn.init.zeros_(enc.bias)
                # Set the x-portion (first dit_dim columns) to identity
                with torch.no_grad():
                    enc.weight[:, :dit_dim].copy_(torch.eye(dit_dim))

        else:
            raise ValueError(f"Unknown injection mode: {mode!r}. Expected 'additive' or 'concat_project'.")

    def inject(self, x: torch.Tensor, track_tokens: torch.Tensor, block_idx: int):
        """Inject track signal into hidden state.

        Args:
            x: (B, L, dit_dim) — hidden state entering this block.
            track_tokens: (B, L, track_dim) — track tokens aligned to full sequence.
            block_idx: which block (0..num_blocks-1).

        Returns:
            x_injected: (B, L, dit_dim) — hidden state with track signal.
        """
        with track4d_record_function(f"track4d360/track_injector/{self.mode}"):
            if self.mode == "additive":
                track_emb = self.encoders[block_idx](track_tokens)
                out = x + track_emb
            elif self.mode == "concat_project":
                out = self.encoders[block_idx](
                    torch.cat([x, track_tokens], dim=-1)
                )
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if os.environ.get("TRACK4D360_DEBUG_TRACK_TOKENS", "0") == "1":
                self._maybe_print_debug(x, track_tokens, out, block_idx)
            return out

    def _maybe_print_debug(
        self,
        x: torch.Tensor,
        track_tokens: torch.Tensor,
        out: torch.Tensor,
        block_idx: int,
    ) -> None:
        debug_blocks = {0, max(self.num_blocks // 2, 0), max(self.num_blocks - 1, 0)}
        if block_idx not in debug_blocks:
            return
        printed = getattr(self, "_track4d_debug_printed_blocks", set())
        if block_idx in printed:
            return
        printed = set(printed)
        printed.add(block_idx)
        self._track4d_debug_printed_blocks = printed

        with torch.no_grad():
            delta = out - x
            x_abs_mean = float(x.detach().abs().mean().cpu())
            track_abs_mean = float(track_tokens.detach().abs().mean().cpu())
            out_abs_mean = float(out.detach().abs().mean().cpu())
            delta_abs_mean = float(delta.detach().abs().mean().cpu())
            delta_abs_max = float(delta.detach().abs().amax().cpu())
        print(
            "[Track4D-360/debug] track_injector "
            f"mode={self.mode} block={block_idx}/{self.num_blocks - 1} "
            f"x_abs_mean={x_abs_mean:.6f} track_abs_mean={track_abs_mean:.6f} "
            f"out_abs_mean={out_abs_mean:.6f} "
            f"delta_abs_mean={delta_abs_mean:.6f} delta_abs_max={delta_abs_max:.6f}",
            flush=True,
        )
