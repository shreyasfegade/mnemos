# Manual check that a real sleep locks the store (the automated test sends a synthetic broadcast).
#
# Run from the repository root, then put the machine to sleep within the countdown and wake it.
# The script reports whether the guard saw PBT_APMSUSPEND and whether the store refused to search
# afterwards.
#
#   powershell -ExecutionPolicy Bypass -File scripts/suspend_test.ps1

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$code = @'
import tempfile, time, pathlib
import numpy as np
from eval.exp2 import lab
from mnemos.power import PowerGuard

rng = np.random.default_rng(0)
cents = rng.standard_normal((16, 64)).astype(np.float32)
cents /= np.linalg.norm(cents, axis=1, keepdims=True)
over = lab.base_overrides(dim=64, buckets=16, capacity=16, block=256, epoch_writes=4)
eng, _, _ = lab.create("m", pathlib.Path(tempfile.mkdtemp()) / "s", cents, over, {})
v = rng.standard_normal(64).astype(np.float32)
v /= np.linalg.norm(v)
eng.put("r0", v.tolist(), "{}")
guard = PowerGuard([eng]).start()
print("Put the machine to sleep now (Start > Power > Sleep), then wake it. Waiting 120 s...")
deadline = time.time() + 120
while time.time() < deadline and guard.suspends == 0:
    time.sleep(1)
guard.stop()
print(f"suspend notifications seen: {guard.suspends}")
try:
    eng.search(v.tolist())
    print("FAIL: the store still searches after the suspend window")
except Exception:
    print("PASS: the store is locked")
'@
Push-Location (Join-Path $PSScriptRoot "..\python")
try { & $py -c $code } finally { Pop-Location }
