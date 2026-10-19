"""Sample released dense examples with 14B or 5B, without cluster data paths."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--example", default="davis_bear")
    parser.add_argument("--model", choices=("14b", "5b-commercial", "5b-full"), default="14b")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/examples")
    parser.add_argument("--validate-only", action="store_true")
    options, passthrough = parser.parse_known_args()
    from holo4d.examples_io import condition_to_sample, load_example, resize_condition, validate_condition

    entry, data, report = load_example(options.manifest, options.example)
    if options.model == "5b-full":
        if report["height"] > report["width"]:
            raise ValueError("The released 704x1280 checkpoint has no validated portrait recipe; choose a landscape example")
        data = resize_condition(data, 704, 1280)
        report = validate_condition(data)
    print(json.dumps({"example": entry["id"], "caption": entry["caption"], **report}), flush=True)
    if options.validate_only:
        return
    from holo4d.fa3_compat import configure_flash_attention_imports
    configure_flash_attention_imports()
    sys.path.insert(0, str(ROOT / "examples/wan2.1_fun"))
    from eval_track4d360_davis_4rc_novel_traj import parse_args
    from eval_track4d360_hybrid_dense_syn4d import build_eval_pipeline, _run_one_pair, precompute_eval_prompt_embs, NEGATIVE_PROMPT
    from holo4d import hub

    kind = "t2v" if options.model == "14b" else options.model
    wan_kind = "t2v" if options.model == "14b" else "5b"
    wan = Path(os.environ["WAN_ROOT"]) if os.environ.get("WAN_ROOT") else hub.download_wan(wan_kind, root=hub.cache_root())
    checkpoint = Path(os.environ["CHECKPOINT"]) if os.environ.get("CHECKPOINT") else hub.download_checkpoint(kind, root=hub.cache_root())
    prefix = hub.WAN_REPOS[wan_kind]
    dit = os.environ.get("DIT_GLOB", str(wan / "diffusion_pytorch_model*.safetensors"))
    vae = os.environ.get("VAE_PATH", str(wan / ("Wan2.1_VAE.pth" if wan_kind == "t2v" else "Wan2.2_VAE.pth")))
    t5 = os.environ.get("T5_PATH", str(wan / "models_t5_umt5-xxl-enc-bf16.pth"))
    tokenizer = os.environ.get("TOKENIZER_PATH", str(wan / "google/umt5-xxl"))
    model_bundle = f"{prefix}:{dit},{prefix}:{vae}"
    if os.environ.get("EVAL_T5_CACHE"):
        cache = Path(os.environ["EVAL_T5_CACHE"])
        if not cache.is_file():
            raise FileNotFoundError(f"Missing EVAL_T5_CACHE: {cache}")
        os.environ["HOLO4D_EVAL_T5_CACHE"] = str(cache)
        os.environ["HOLO4D_REQUIRE_T5_CACHE"] = "1"
        os.environ["HOLO4D_ALLOW_ONLINE_T5_ENCODE"] = "0"
    else:
        model_bundle += f",{prefix}:{t5}"
        os.environ.setdefault("HOLO4D_ALLOW_ONLINE_T5_ENCODE", "1")
    config = "persistent_identity_target_srcdepth_srcrgb.yaml" if options.model == "5b-full" else "default_conv3d_patchify_srcdepth_srcrgb.yaml"
    sys.argv = [sys.argv[0], "--checkpoint_path", str(checkpoint),
                "--output_dir", str(options.output_dir),
                "--camera_control_checkpoint", dit,
                "--model_id_with_origin_paths", model_bundle,
                "--tokenizer_path", tokenizer, "--track_config", str(ROOT / "config/holo4d" / config),
                "--height", str(report["height"]), "--width", str(report["width"]), "--num_frames", "49",
                "--no_dense_branch", "--use_query_frame_impulse_condition", "--use_plucker_camera_control",
                "--enable_v2v_plucker_camera_control", "--use_perlayer_camera_control",
                "--camera_injection_mode", "projector", "--camera_plucker_encoder_mode", "patchify",
                "--use_plucker_raymap_tokens", "--track_injection_mode", "per_block",
                "--track_injection_block_mode", "concat_project", "--track_spatial_stride", "1",
                "--num_inference_steps", "50", "--cfg_scale", "5", "--sigma_shift", "5", "--seed", "42", *passthrough]
    args = parse_args()
    if (args.height, args.width) != (report["height"], report["width"]):
        raise ValueError("Sampling canvas must match the prepared condition; resize RGB/depth and all K together first")
    if options.model == "5b-full" and (args.height, args.width) != (704, 1280):
        raise ValueError("5b-full was trained at 704x1280; the image and intrinsics must use that canvas, not 384p.")
    args.skip_comparison = True
    args.vae_tiling = True
    args.capture_final_latents = False
    output = options.output_dir / options.model / options.example
    output.mkdir(parents=True, exist_ok=True)
    contract = {"model": options.model, "checkpoint": str(checkpoint), "example": entry,
                "steps": args.num_inference_steps, "cfg": args.cfg_scale, "shift": args.sigma_shift,
                "seed": args.seed, "fps": 16, "mp4_quality": 9, **report}
    if (output / "COMPLETE.json").is_file() and (output / "pred_rgb.mp4").is_file():
        if json.loads((output / "COMPLETE.json").read_text()) == contract:
            print(f"[examples] skip verified complete {output}")
            return
        raise ValueError("Existing output has a different contract; choose a new output directory")
    if (output / "pred_rgb.mp4").exists():
        raise ValueError("Incomplete existing prediction: choose a new output directory; no overwrite")
    (output / "run_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    pipe = build_eval_pipeline(args)
    precompute_eval_prompt_embs(pipe, [entry["caption"], NEGATIVE_PROMPT])
    _run_one_pair(pipe, condition_to_sample(data, entry), args, output)
    (output / "COMPLETE.json").write_text(json.dumps(contract, indent=2) + "\n")
    print(f"[examples] complete {output / 'pred_rgb.mp4'}", flush=True)


if __name__ == "__main__":
    main()
