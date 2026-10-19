#!/bin/bash
# Resolve portable VGGT-Omega and CowTracker paths for Track4D eval launchers.

track4d_first_existing_dir() {
  local fallback="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [ -d "${candidate}" ]; then
      printf '%s' "${candidate}"
      return
    fi
  done
  printf '%s' "${fallback}"
}

track4d_first_existing_file() {
  local fallback="$1"
  shift
  local candidate
  for candidate in "$@"; do
    if [ -f "${candidate}" ]; then
      printf '%s' "${candidate}"
      return
    fi
  done
  printf '%s' "${fallback}"
}

track4d_configure_external_methods() {
  local repo_root="$1"
  local repo_parent
  repo_parent="$(dirname "${repo_root}")"

  VGGT_OMEGA_REPO="${VGGT_OMEGA_REPO:-$(track4d_first_existing_dir \
    "${repo_root}/external/vggt-omega" \
    "${repo_parent}/baselines/vggt-omega" \
    "${repo_parent}/vggt-omega")}"
  COWTRACKER_REPO="${COWTRACKER_REPO:-$(track4d_first_existing_dir \
    "${repo_root}/external/cowtracker" \
    "${repo_parent}/baselines/cowtracker" \
    "${repo_parent}/cowtracker")}"
  VGGT_OMEGA_CHECKPOINT="${VGGT_OMEGA_CHECKPOINT:-$(track4d_first_existing_file \
    "${repo_root}/weights/VGGT-Omega/vggt_omega_1b_512.pt" \
    "${repo_parent}/weights/VGGT-Omega/vggt_omega_1b_512.pt")}"

  if [ -z "${COWTRACKER_CHECKPOINT:-}" ] && \
     [ -f "${repo_root}/weights/CoWTracker/cowtracker_model.pth" ]; then
    COWTRACKER_CHECKPOINT="${repo_root}/weights/CoWTracker/cowtracker_model.pth"
  fi
  export VGGT_OMEGA_REPO COWTRACKER_REPO VGGT_OMEGA_CHECKPOINT
  export COWTRACKER_CHECKPOINT="${COWTRACKER_CHECKPOINT:-}"
}
