# shellcheck shell=bash
# -----------------------------------------------------------------------------
# Resolve real libcuda / libnvidia-ml in containers (stub vs bind-mounted driver).
# Fixes Triton / vLLM: undefined symbol: cuModuleGetFunction
#
# Source before launching Python, e.g.:
#   source "$(dirname "$0")/nvidia_driver_preload.inc.sh"
# -----------------------------------------------------------------------------

if [ -d /usr/local/nvidia/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

_NV_FIX_DIR="/tmp/nvidia-runtime-fix"
rm -rf "$_NV_FIX_DIR" && mkdir -p "$_NV_FIX_DIR"
_LIBCUDA_REAL=""
_LIBNVML_REAL=""
_NV_SEARCH="/usr/local/nvidia/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /usr/local/lib"
echo "[nvidia-detect] Searching for real NVIDIA libraries on $(hostname)..."

for _lib in libcuda libnvidia-ml; do
  _found=""

  for _d in $_NV_SEARCH; do
    _f="$_d/${_lib}.so.1"
    [ -e "$_f" ] || continue
    _sz=$(stat -L -c%s "$_f" 2>/dev/null || echo 0)
    if [ "$_sz" -gt 1024 ]; then
      _found="$_f"
      echo "[nvidia-detect] M1 ✓ ${_lib}.so.1 at $_f (${_sz}B, stat -L)"
      break
    fi
  done

  if [ -z "$_found" ]; then
    for _d in $_NV_SEARCH; do
      [ -d "$_d" ] || continue
      for _f in "$_d"/${_lib}.so.[0-9]*; do
        [ -e "$_f" ] || continue
        _sz=$(stat -L -c%s "$_f" 2>/dev/null || echo 0)
        if [ "$_sz" -gt 1024 ]; then
          _found="$_f"
          echo "[nvidia-detect] M2 ✓ ${_lib} versioned at $_f (${_sz}B)"
          break 2
        fi
      done
    done
  fi

  if [ -z "$_found" ]; then
    _lp=$(/sbin/ldconfig -p 2>/dev/null | awk "/${_lib}\\.so\\.1 /{print \$NF; exit}")
    if [ -n "$_lp" ] && [ -e "$_lp" ]; then
      _sz=$(stat -L -c%s "$_lp" 2>/dev/null || echo 0)
      if [ "$_sz" -gt 1024 ]; then
        _found="$_lp"
        echo "[nvidia-detect] M3 ✓ ${_lib} via ldconfig: $_lp (${_sz}B)"
      fi
    fi
  fi

  if [ -z "$_found" ] && command -v nvidia-smi >/dev/null 2>&1; then
    _lp=$(ldd "$(which nvidia-smi)" 2>/dev/null | awk "/${_lib}/{print \$3; exit}")
    if [ -n "$_lp" ] && [ -e "$_lp" ]; then
      _sz=$(stat -L -c%s "$_lp" 2>/dev/null || echo 0)
      if [ "$_sz" -gt 1024 ]; then
        _found="$_lp"
        echo "[nvidia-detect] M4 ✓ ${_lib} via ldd nvidia-smi: $_lp (${_sz}B)"
      fi
    fi
  fi

  if [ -z "$_found" ]; then
    _found=$(timeout 5 find /usr /lib /run -maxdepth 5 -name "${_lib}.so.1" -exec sh -c '
      _sz=$(stat -L -c%s "$1" 2>/dev/null || echo 0)
      [ "$_sz" -gt 1024 ] && echo "$1"
    ' _ {} \; 2>/dev/null | head -1)
    [ -n "$_found" ] && echo "[nvidia-detect] M5 ✓ ${_lib} via broad find: $_found"
  fi

  if [ -n "$_found" ]; then
    [ "$_lib" = "libcuda" ]      && _LIBCUDA_REAL="$_found"
    [ "$_lib" = "libnvidia-ml" ] && _LIBNVML_REAL="$_found"
    ln -sf "$_found" "$_NV_FIX_DIR/${_lib}.so.1"
    ln -sf "$_found" "$_NV_FIX_DIR/${_lib}.so"
  else
    echo "[nvidia-detect] ✗ ${_lib}: NOT FOUND (all candidates < 1KB)"
  fi
done

if [ -n "$_LIBCUDA_REAL" ]; then
  export TRITON_LIBCUDA_PATH="$_NV_FIX_DIR"
  export LD_PRELOAD="${_LIBCUDA_REAL}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [ -n "$_LIBNVML_REAL" ]; then
  export LD_PRELOAD="${_LIBNVML_REAL}${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if [ -e "$_NV_FIX_DIR/libcuda.so.1" ]; then
  export LD_LIBRARY_PATH="${_NV_FIX_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
echo "[nvidia-detect] LD_PRELOAD=${LD_PRELOAD:-<unset>}"
echo "[nvidia-detect] LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo "[nvidia-detect] TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH:-<unset>}"
