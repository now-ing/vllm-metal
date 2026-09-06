#!/bin/bash
# Common library functions for vllm-metal scripts

# Print an error message
error() {
  echo -e "Error: $*" >&2
}

# Print a warning message
warning() {
  echo -e "Warning: $*" >&2
}

# Print a success message
success() {
  echo -e "✓ $*"
}

# Print a section header
section() {
  echo "=== $* ==="
}

validate_vllm_release_tag() {
  local release_tag="$1"
  if [[ ! "$release_tag" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
    error "Invalid vLLM release tag: ${release_tag}"
    return 1
  fi
}

# Read the upstream vLLM release paired with this source revision.
read_vllm_release_tag() {
  local path="${1:-.github/vllm-release-tag.commit}"
  local release_tag

  if [ ! -r "$path" ]; then
    error "Missing vLLM release tag file: ${path}"
    return 1
  fi

  release_tag=$(tr -d '[:space:]' < "$path")
  validate_vllm_release_tag "$release_tag" || return 1
  printf '%s\n' "$release_tag"
}

# Check if running on Apple Silicon
is_apple_silicon() {
  [ "$(uname -m)" = "arm64" ]
}

# Require a native arm64 Python interpreter.
require_arm64_python() {
  local python_bin="${1:-python}"
  local machine

  if ! machine=$("$python_bin" -c "import platform; print(platform.machine())"); then
    error "Failed to inspect Python architecture using ${python_bin}."
    return 1
  fi

  if [ "$machine" != "arm64" ]; then
    error "vllm-metal requires native arm64 Python, got ${machine}. Remove the venv and rerun install.sh from an arm64 Python."
    return 1
  fi
}

# Ensure uv is installed
ensure_uv() {
  if ! command -v uv &> /dev/null; then
    echo "uv not found, installing..."
    if ! curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh; then
      error "Failed to install uv"
      return 1
    fi

    # Add uv to PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

# Ensure virtual environment exists and is activated
ensure_venv() {
  if [ ! -d "$1" ]; then
    section "Creating virtual environment"
    uv venv "$1" --clear --python 3.12 --seed
  fi

  # shellcheck source=/dev/null
  source "$1/bin/activate"
}

# Install dev dependencies
install_dev_deps() {
  section "Installing dependencies"
  uv pip install -e ".[dev]"
}

# Full development environment setup
setup_dev_env() {
  ensure_uv
  ensure_venv ".venv-vllm-metal"
  install_dev_deps
}

# Get version from pyproject.toml
get_version() {
  uv run python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
}

# Ensure `xcrun metal` can actually compile a .metallib.
#
# Producing a .metallib needs the Metal toolchain. On Xcode 26+ it is a
# separate downloadable component; older Xcode bundles it. Rather than guess,
# trial-compile a trivial shader: if that succeeds the toolchain is already
# present (no slow download), otherwise download MetalToolchain and re-check.
ensure_metal_toolchain() {
  section "Ensuring Metal toolchain"
  # Keep Maturin/Rust aligned with the native extension's macOS 15 floor.
  export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"

  local tmpdir metal_src metal_lib
  tmpdir=$(mktemp -d)
  metal_src="${tmpdir}/probe.metal"
  metal_lib="${tmpdir}/probe.metallib"
  printf '[[kernel]] void _t() {}\n' > "${metal_src}"

  if xcrun -sdk macosx metal -o "${metal_lib}" "${metal_src}" &> /dev/null; then
    success "Metal toolchain present"
    rm -rf "${tmpdir}"
    return 0
  fi

  echo "Metal toolchain not available, downloading via xcodebuild..."
  # A Command Line Tools-only machine has no xcodebuild at all; treat that as
  # "toolchain unavailable" rather than letting the non-zero status abort the
  # caller (install.sh falls back to prebuilt or JIT artifacts).
  if ! xcodebuild -downloadComponent MetalToolchain; then
    warning "Could not download the Metal toolchain (xcodebuild failed or unavailable)."
    rm -rf "${tmpdir}"
    return 1
  fi

  if ! xcrun -sdk macosx metal -o "${metal_lib}" "${metal_src}" &> /dev/null; then
    error "Metal toolchain still unavailable after download; cannot compile .metallib."
    rm -rf "${tmpdir}"
    return 1
  fi

  success "Metal toolchain ready"
  rm -rf "${tmpdir}"
}

# Build the in-package native artifacts (the _paged_ops*.so and the required
# precompiled .metallib shader libraries, including NAX) into vllm_metal/metal/
# so `uv build` can bundle them via the maturin `include` directive.
#
# `python` here is the venv interpreter activated by setup_dev_env, so mlx and
# nanobind are importable.
build_native_artifacts() {
  section "Building native Metal artifacts"
  # Official wheels require NAX, so reject an older SDK before compiling.
  if ! python -c \
    "from vllm_metal.metal.build import require_nax_sdk; require_nax_sdk()"; then
    return 1
  fi
  python -m vllm_metal.metal.build
}

# Populate vllm_metal/metal/ with the prebuilt native artifacts (the
# _paged_ops*.so extension and the .metallib shader libraries, including NAX)
# unpacked from the newest release wheel whose tag matches this checkout's
# pyproject version. This is the fallback for machines without the Metal
# toolchain (e.g. Xcode Command Line Tools only), where .metallib files
# cannot be compiled locally. The runtime loads these artifacts as-is, so
# an editable Python install plus the unpacked files is fully functional.
#
# Must run from the repo root with the venv active (install.sh source-checkout
# path guarantees both), because the expected file names are resolved from the
# locally checked-out vllm_metal.metal.build.
fetch_prebuilt_native_artifacts() {
  section "Fetching prebuilt native artifacts from release wheel"

  # The venv python is already active here; reading pyproject through it
  # avoids the noisy `uv run` VIRTUAL_ENV-mismatch warning.
  local version
  if ! version=$(python -c \
      "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"); then
    error "Failed to read the project version from pyproject.toml."
    return 1
  fi

  # Match vX.Y.Z (stable) or vX.Y.Z.devN (dev channel) releases only, so the
  # artifacts' ABI stays in family with this checkout.
  local release_data
  if ! release_data=$(curl -fsSL \
      "https://api.github.com/repos/vllm-project/vllm-metal/releases?per_page=30"); then
    warning "Failed to query GitHub releases for prebuilt artifacts."
    return 1
  fi

  local wheel_url
  wheel_url=$(VERSION="${version}" RELEASE_DATA="${release_data}" python3 -c '
import json
import os
import re
import sys

try:
    releases = json.loads(os.environ["RELEASE_DATA"])
except Exception:
    sys.exit(0)

# /releases is newest-first; take the newest wheel within this version family.
pattern = re.compile(r"^v" + re.escape(os.environ["VERSION"]) + r"(\.dev[0-9]+)?$")
for release in releases:
    if not pattern.match(release.get("tag_name") or ""):
        continue
    for asset in release.get("assets", []):
        if (asset.get("name") or "").endswith(".whl"):
            print(asset.get("browser_download_url", ""))
            sys.exit(0)
')

  if [[ -z "${wheel_url}" ]]; then
    warning "No release wheel matches version ${version}."
    return 1
  fi

  local tmp_dir wheel_path
  tmp_dir=$(mktemp -d)
  register_cleanup_dir "${tmp_dir}"
  wheel_path="${tmp_dir}/$(basename "${wheel_url}")"
  echo "Downloading ${wheel_path##*/}..."
  if ! curl -fsSL "${wheel_url}" -o "${wheel_path}"; then
    warning "Failed to download the release wheel."
    return 1
  fi

  # Unpack only the prebuilt artifacts into the source tree; build-time
  # .sha256 sidecars are not needed because the runtime loader does not
  # consult them.
  if ! WHEEL_PATH="${wheel_path}" python -c "
import os
import sys
import zipfile

from vllm_metal.metal.build import (
    METALLIB_NAMES,
    NAX_METALLIB_NAME,
    metallib_path,
    output_path,
)

prefix = 'vllm_metal/metal/'
names = {output_path().name}
names.update(metallib_path(n).name for n in (*METALLIB_NAMES, NAX_METALLIB_NAME))
with zipfile.ZipFile(os.environ['WHEEL_PATH']) as zf:
    members = [m for m in zf.namelist() if m.startswith(prefix) and m[len(prefix):] in names]
    missing = names - {m[len(prefix):] for m in members}
    if missing:
        print('wheel is missing artifacts: ' + ', '.join(sorted(missing)), file=sys.stderr)
        sys.exit(1)
    zf.extractall('.', members=members)
print('unpacked %d artifacts' % len(members))
"; then
    warning "The release wheel does not carry the expected native artifacts."
    return 1
  fi

  success "Prebuilt native artifacts installed into vllm_metal/metal/"
}

# Fail unless the freshly built wheel actually bundles the prebuilt native
# artifacts: the _paged_ops*.so extension, three required metallibs, and NAX.
# maturin's `include` directive is what pulls these (gitignored)
# files in; if that ever regresses, the wheel would install fine but fail at
# first run with "Prebuilt native extension not found". The expected filenames
# are read from build.py so this guard never drifts from the runtime loader.
#
# Usage: verify_wheel_artifacts <path-to-wheel>
verify_wheel_artifacts() {
  local wheel="$1"
  section "Verifying wheel bundles native artifacts"

  local expected
  if ! expected=$(python -c "
from vllm_metal.metal.build import METALLIB_NAMES, NAX_METALLIB_NAME, metallib_path, output_path
print(output_path().name)
for _name in (*METALLIB_NAMES, NAX_METALLIB_NAME):
    print(metallib_path(_name).name)
"); then
    error "Failed to resolve expected native artifact names from build.py."
    return 1
  fi

  local contents name
  contents=$(unzip -l "$wheel")
  while IFS= read -r name; do
    [ -z "$name" ] && continue
    if grep -qF "$name" <<< "$contents"; then
      success "bundled: ${name}"
    else
      error "Wheel ${wheel} is missing native artifact: ${name}"
      error "maturin [tool.maturin] 'include' likely failed to bundle it."
      return 1
    fi
  done <<< "$expected"

  local expected_minos paged_ops_name unpack_dir native_so native_name actual_minos native_count
  expected_minos=$(python -c "from vllm_metal.metal.build import MIN_MACOS_VERSION; print(MIN_MACOS_VERSION)")
  paged_ops_name=$(python -c "from vllm_metal.metal.build import output_path; print(output_path().name)")
  unpack_dir=$(mktemp -d)
  unzip -qq "${wheel}" '*.so' -d "${unpack_dir}"
  native_count=0
  while IFS= read -r native_so; do
    native_name=$(basename "${native_so}")
    case "${native_name}" in
      "${paged_ops_name}"|_rs.*.so) ;;
      *) continue ;;
    esac
    native_count=$((native_count + 1))
    actual_minos=$(otool -l "${native_so}" | awk '$1 == "minos" { print $2; exit }')
    if [ "${actual_minos}" != "${expected_minos}" ]; then
      error "${native_so} targets macOS ${actual_minos:-unknown}; expected ${expected_minos}."
      rm -rf "${unpack_dir}"
      return 1
    fi
    success "${native_name}: macOS ${actual_minos}"
  done < <(find "${unpack_dir}" -type f -name '*.so')
  rm -rf "${unpack_dir}"
  if [ "${native_count}" -lt 2 ]; then
    error "Wheel ${wheel} is missing a required native extension."
    return 1
  fi

  success "Wheel bundles all native artifacts"
}
