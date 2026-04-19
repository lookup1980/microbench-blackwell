#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ENV_DIR="${ROOT_DIR}/.repo_env"
MAMBA_ROOT_PREFIX="${REPO_ENV_DIR}/micromamba"
ENV_NAME="${ENV_NAME:-cuda-dev}"
ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
CUDA_VERSION="${CUDA_VERSION:-13.1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
MICROMAMBA_BIN="${REPO_ENV_DIR}/bin/micromamba"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--env-name NAME] [--cuda-version X.Y] [--python-version X.Y]

Creates a repo-local CUDA development environment under:
  ${REPO_ENV_DIR}

Notes:
  - This does not install the NVIDIA driver.
  - Runtime GPU access still depends on the host driver/kernel stack.
  - Nsight Compute is expected from the host install if you want profiler runs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      ENV_PREFIX="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}"
      shift 2
      ;;
    --cuda-version)
      CUDA_VERSION="$2"
      shift 2
      ;;
    --python-version)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

mkdir -p "${REPO_ENV_DIR}/bin"

download() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl -LfsS "$url" -o "$out"
    return
  fi

  if command -v wget >/dev/null 2>&1; then
    wget -qO "$out" "$url"
    return
  fi

  echo "Need curl or wget to download micromamba." >&2
  exit 1
}

bootstrap_micromamba() {
  if [[ -x "${MICROMAMBA_BIN}" ]]; then
    return
  fi

  local archive
  archive="$(mktemp)"

  echo "Bootstrapping micromamba into ${MICROMAMBA_BIN}"
  download "https://micro.mamba.pm/api/micromamba/linux-64/latest" "${archive}"
  tar -xjf "${archive}" -C "${REPO_ENV_DIR}"
  rm -f "${archive}"

  if [[ ! -x "${REPO_ENV_DIR}/bin/micromamba" ]]; then
    echo "micromamba bootstrap failed." >&2
    exit 1
  fi
}

create_env() {
  local -a packages=(
    "python=${PYTHON_VERSION}"
    "pip"
    "numpy"
    "pandas"
    "matplotlib"
    "scipy"
    "jupyterlab"
    "cmake"
    "ninja"
    "make"
    "pkg-config"
    "git"
    "cxx-compiler"
    "cuda-version=${CUDA_VERSION}"
    "cuda-nvcc"
    "cuda-cudart-dev"
    "cuda-profiler-api"
    "cuda-nvtx"
    "cuda-cccl"
    "cuda-libraries-dev"
  )

  echo "Creating local environment at ${ENV_PREFIX}"
  MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}" "${MICROMAMBA_BIN}" create -y \
    -n "${ENV_NAME}" \
    -c conda-forge \
    -c nvidia \
    "${packages[@]}"
}

write_activate_helpers() {
  mkdir -p "${REPO_ENV_DIR}"

  cat > "${REPO_ENV_DIR}/activate.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX}"
MICROMAMBA_BIN="${MICROMAMBA_BIN}"
ENV_NAME="${ENV_NAME}"

export MAMBA_ROOT_PREFIX="\${MAMBA_ROOT_PREFIX}"
eval "\$("\${MICROMAMBA_BIN}" shell hook --shell bash)"
set +u
micromamba activate "\${ENV_NAME}"
set -u

export CUDA_HOME="\${CONDA_PREFIX}"
export CUDA_PATH="\${CONDA_PREFIX}"
export PATH="\${CONDA_PREFIX}/bin:\${PATH}"
export LD_LIBRARY_PATH="\${CONDA_PREFIX}/lib:\${CONDA_PREFIX}/lib64:\${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="\${CONDA_PREFIX}:\${CMAKE_PREFIX_PATH:-}"

if command -v ncu >/dev/null 2>&1; then
  echo "Using host ncu: \$(command -v ncu)"
else
  echo "ncu not found on PATH; profiler-backed benchmark scripts may fail." >&2
fi

echo "Activated local env: \${CONDA_PREFIX}"
echo "CUDA_HOME=\${CUDA_HOME}"
echo "nvcc=\$(command -v nvcc)"
EOF

  chmod +x "${REPO_ENV_DIR}/activate.sh"
}

print_summary() {
  cat <<EOF

Local environment ready.

Activate it with:
  source "${REPO_ENV_DIR}/activate.sh"

Installed into:
  ${ENV_PREFIX}

Expected from the host:
  - NVIDIA kernel driver
  - GPU device access
  - Nsight Compute (optional, but required by some benchmark scripts)

Suggested validation after activation:
  nvcc --version
  python --version
  ncu --version || true
EOF
}

bootstrap_micromamba
create_env
write_activate_helpers
print_summary
