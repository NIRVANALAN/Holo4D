#!/bin/bash
# Shared DAVIS benchmark asset defaults for Track4D-360 eval wrappers.
#
# Prefer the historical ${BASELINES_ROOT}/_baselines_env.sh when available.
# On remote systems where only the needed assets are synced, fall back to
# explicit/default paths and let callers validate the assets they need.

benchmark_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
benchmark_eval_dir="$(cd "${benchmark_env_dir}/.." && pwd)"
# This public copy lives at eval/common (not bash_scripts/.../eval/common).
benchmark_repo_root="$(cd "${benchmark_env_dir}/../.." && pwd)"

export BASELINES_ROOT="${BASELINES_ROOT:-${benchmark_repo_root}/outputs/benchmarks}"

# Resolve a benchmark asset that may live under ${BASELINES_ROOT} or one level
# up in the benchmark root. Some clusters only populate the benchmark root (the
# layout the sampler reads/writes davis_trajectories.csv, cameras_2traj, etc.).
# Prefer the baselines copy when present, else fall back to the benchmark root so
# metrics and sampling always resolve the same asset.
_resolve_baseline_asset() {
  local rel="$1"
  if [ -e "${BASELINES_ROOT}/${rel}" ]; then
    printf '%s' "${BASELINES_ROOT}/${rel}"
  else
    printf '%s' "$(dirname "${BASELINES_ROOT}")/${rel}"
  fi
}

if [ -f "${BASELINES_ROOT}/_baselines_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${BASELINES_ROOT}/_baselines_env.sh"
else
  export BASELINES_REPO_ROOT="${BASELINES_REPO_ROOT:-${benchmark_repo_root}/external}"
  export HOLO4D_ROOT="${HOLO4D_ROOT:-${benchmark_repo_root}}"
  case ":${PYTHONPATH:-}:" in
    *":${HOLO4D_ROOT}:"*) ;;
    *) export PYTHONPATH="${HOLO4D_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
  export DAVIS_ROOT="${DAVIS_ROOT:-$(_resolve_baseline_asset davis_video)}"
  export CAPTION_JSON="${CAPTION_JSON:-$(_resolve_baseline_asset davis_captions.json)}"
  export SEQUENCE_LIST_PATH="${SEQUENCE_LIST_PATH:-$(_resolve_baseline_asset davis_sequences.txt)}"
  export PAIRS_CSV="${PAIRS_CSV:-$(_resolve_baseline_asset davis_trajectories.csv)}"
  export PAIRS_FULL_CSV="${PAIRS_FULL_CSV:-$(_resolve_baseline_asset davis_trajectories_full.csv)}"
  export CAMERAS_ROOT="${CAMERAS_ROOT:-$(_resolve_baseline_asset cameras_2traj)}"
  export DAVIS_4RC_ROOT="${DAVIS_4RC_ROOT:-$(_resolve_baseline_asset 4RC/davis_49f)}"
  export NUM_FRAMES="${NUM_FRAMES:-49}"
fi

export PAIRS_FULL_CSV="${PAIRS_FULL_CSV:-${BASELINES_ROOT}/davis_trajectories_full.csv}"
export CAMERAS_ROOT="${CAMERAS_ROOT:-${BASELINES_ROOT}/cameras_2traj}"

ensure_ffmpeg_on_path() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi

  local ffmpeg_exe=""
  ffmpeg_exe="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
  if [ -n "${ffmpeg_exe}" ] && [ -x "${ffmpeg_exe}" ]; then
    local ffmpeg_link_dir="${TMPDIR:-/tmp}/track4d360_ffmpeg_bin"
    mkdir -p "${ffmpeg_link_dir}"
    ln -sf "${ffmpeg_exe}" "${ffmpeg_link_dir}/ffmpeg"
    export PATH="${ffmpeg_link_dir}:${PATH}"
    return 0
  fi

  local candidate
  for candidate in /usr/bin/ffmpeg /usr/local/bin/ffmpeg; do
    if [ -x "${candidate}" ]; then
      export PATH="$(dirname "${candidate}"):${PATH}"
      return 0
    fi
  done

  echo "ERROR: ffmpeg not found in PATH and imageio_ffmpeg fallback is unavailable." >&2
  echo "Install/sync ffmpeg or ensure imageio_ffmpeg is installed in the active metrics conda env." >&2
  return 1
}
holo4d_activate_conda() {
  local env_name="$1"
  if [ -n "${CONDA_SH:-}" ]; then
    if [ ! -f "${CONDA_SH}" ]; then
      echo "ERROR: CONDA_SH does not exist: ${CONDA_SH}" >&2
      return 1
    fi
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
  elif ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available; activate ${env_name} first or set CONDA_SH." >&2
    return 1
  elif [ "$(type -t conda)" != "function" ]; then
    eval "$(conda shell.bash hook)"
  fi
  if [ -n "${CONDA_DEFAULT_ENV:-}" ] && [ "${CONDA_DEFAULT_ENV}" = "${env_name}" ]; then
    return 0
  fi
  conda activate "${env_name}"
}
