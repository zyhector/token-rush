#!/usr/bin/env bash
# "Is this box usable?" — run once on every newly rented instance.
#
# Dumps machine identity (vast machine/host id, GPU, CPU, storage, toolchain),
# then checks which NVIDIA profilers actually work here. The profiler result is
# a rental accept/reject criterion: ncu counter access is a per-HOST setting.
#
# Pair with check_bandwidth.py, which re-anchors the decode roofline for this card.
set -uo pipefail

hdr() { printf '\n=== %s ===\n' "$1"; }

hdr "VAST INSTANCE"
if [ -n "${CONTAINER_API_KEY:-}" ] && command -v vastai >/dev/null; then
  vastai show instance "${CONTAINER_ID}" --api-key "${CONTAINER_API_KEY}" --raw 2>/dev/null |
    python3 -c "
import json,sys
d=json.load(sys.stdin)
for k in ['id','machine_id','host_id','geolocation','mobo_name','dph_total',
          'gpu_name','gpu_ram','gpu_mem_bw','pcie_bw','cpu_name','cpu_cores',
          'cpu_cores_effective','cpu_ram','disk_space','disk_bw','inet_down','inet_up',
          'driver_version','cuda_max_good','compute_cap','image_uuid','reliability2']:
    if k in d: print(f'{k:22s} {d[k]}')
"
else
  echo "CONTAINER_ID=${CONTAINER_ID:-?}  (no CONTAINER_API_KEY; skipping vast API)"
fi

hdr "GPU"
nvidia-smi --query-gpu=name,uuid,vbios_version,pci.bus_id,pcie.link.gen.max,\
pcie.link.width.max,memory.total,clocks.max.sm,clocks.max.mem,power.max_limit,\
compute_cap,driver_version --format=csv,noheader |
  tr ',' '\n' | sed 's/^ *//' |
  paste -d'\t' <(printf '%s\n' name uuid vbios pci_bus pcie_gen pcie_width \
                 vram sm_clock_max mem_clock_max power_limit compute_cap driver) -

hdr "GPU THROTTLE STATE"
nvidia-smi -q -d PERFORMANCE 2>/dev/null | grep -E "Slowdown|Active" | head -8

hdr "CPU"
lscpu | grep -E "^Architecture|^CPU\(s\)|^Model name|^Thread\(s\)|^Core\(s\)|^Socket\(s\)|^NUMA node\(s\)|^CPU max|^L3 cache" | sed 's/  */ /g'

hdr "MEMORY"
free -g | head -2

hdr "STORAGE"
df -h / "${WORKSPACE:-/workspace}" 2>/dev/null
if command -v vast-capabilities >/dev/null; then
  printf 'workspace_is_volume: '
  vast-capabilities 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['instance']['workspace_is_volume'])" 2>/dev/null || echo '?'
fi

hdr "TOOLCHAIN"
printf '%-16s %s\n' nvcc "$(nvcc --version 2>/dev/null | tail -1)"
printf '%-16s %s\n' gcc  "$(gcc --version 2>/dev/null | head -1)"
printf '%-16s %s\n' cmake "$(cmake --version 2>/dev/null | head -1)"
python - <<'PY' 2>/dev/null
import importlib.metadata as md
for p in ['torch','triton','transformers','flash-linear-attention','safetensors',
          'huggingface-hub','numpy','einops','accelerate']:
    try: print(f'{p:16s} {md.version(p)}')
    except Exception: print(f'{p:16s} -')
PY

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

hdr "CAPABILITIES"
grep CapEff /proc/self/status
if command -v capsh >/dev/null; then
  capsh --decode="$(awk '/CapEff/{print $2}' /proc/self/status)" 2>/dev/null |
    tr ',' '\n' | grep -iE 'sys_admin|perfmon' || echo "(no cap_sys_admin / cap_perfmon)"
fi

hdr "NSYS"
# Pick the newest installed nsys. Both 2024.6.2 (CUDA 12.8 repo) and 2025.1.3
# trace sm_120 kernels correctly; either is fine.
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/target-linux-x64/nsys 2>/dev/null | sort -V | tail -1)
NSYS=${NSYS:-$(command -v nsys || true)}
if [ -z "$NSYS" ]; then
  echo "MISSING — apt-get install -y nsight-systems-2025.1.3"
else
  echo "$NSYS"; "$NSYS" --version
  cat > "$TMP/gpu_probe.py" <<'PY'
import torch
x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
for _ in range(10):
    x = torch.nn.functional.silu(x @ x) * 1e-3
torch.cuda.synchronize()
PY
  # NB: never name a probe script after a stdlib module (nt.py, os.py, ...) —
  # the script dir goes on sys.path first, the interpreter dies before CUDA init,
  # and nsys records a trace with no GPU work in it. An empty trace means "the
  # target never ran" at least as often as it means "the profiler failed".
  "$NSYS" profile -t cuda -o "$TMP/trace" --force-overwrite true \
      python "$TMP/gpu_probe.py" >/dev/null 2>&1
  if "$NSYS" stats --report cuda_gpu_kern_sum --force-export true "$TMP/trace.nsys-rep" 2>&1 |
       grep -q "SKIPPED"; then
    echo "RESULT: no kernel data captured — check the probe program actually ran"
  else
    echo "RESULT: kernel timeline captured OK"
  fi
fi

hdr "NCU"
if ! command -v ncu >/dev/null; then
  echo "MISSING"
else
  ncu --version | head -1
  cat > "$TMP/k.cu" <<'CU'
__global__ void k(float* o) { o[threadIdx.x] = threadIdx.x; }
int main() { float* d; cudaMalloc(&d, 512); k<<<1, 32>>>(d); cudaDeviceSynchronize(); }
CU
  nvcc -arch=sm_120 "$TMP/k.cu" -o "$TMP/k" 2>/dev/null
  out=$(ncu --metrics dram__bytes.sum "$TMP/k" 2>&1)
  if grep -q ERR_NVGPUCTRPERM <<<"$out"; then
    echo "RESULT: BLOCKED — ERR_NVGPUCTRPERM (host must set"
    echo "        NVreg_RestrictProfilingToAdminUsers=0; not fixable in-container)"
  else
    echo "RESULT: counters readable"
  fi
fi
