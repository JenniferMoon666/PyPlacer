SRC_DIR = r"D:\Codes\VLSI\PyPlacer_v3_freeze"
OUT_DIR_DEFAULT = r"E:\Workbuddy AI\Paper Use\_runs"

with open(SRC_DIR + r"\main_v3_freeze.py", encoding="utf-8") as f:
    src = f.read()

header = (
    'import sys as _sys\n'
    '_sys.path.insert(0, r"' + SRC_DIR + '")\n'
    'import os as _os\n'
    '_TS   = float(_os.environ.get("TIME_SCALE", "1.0"))\n'
    '_SEED = int(_os.environ.get("SEED", "42"))\n'
    '_OUT  = _os.environ.get("OUT_DIR", r"' + OUT_DIR_DEFAULT + '")\n'
)
src = header + src

REPLACEMENTS = [
    ("dp1_time = 1200 if is_small else (5400 if is_medium else 5400)",
     "dp1_time = int((1200 if is_small else (5400 if is_medium else 5400)) * _TS)"),
    ("dp2_time = 600 if is_small else (1800 if is_medium else 1800)",
     "dp2_time = int((600 if is_small else (1800 if is_medium else 1800)) * _TS)"),
    ("dp3_time = 300 if is_small else (900 if is_medium else 900)",
     "dp3_time = int((300 if is_small else (900 if is_medium else 900)) * _TS)"),
    ("dp = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                      sa_enabled=True, use_centroid_guidance=True)",
     "dp = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                      sa_enabled=True, use_centroid_guidance=True, seed=_SEED)"),
    ("dp2 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                       sa_enabled=True, use_centroid_guidance=True)",
     "dp2 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                       sa_enabled=True, use_centroid_guidance=True, seed=_SEED+1)"),
    ("dp3 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                       sa_enabled=True, use_centroid_guidance=True)",
     "dp3 = WindowDP(bm, device=device, greedy_only=False, gpu_tracker=gpu_tracker,\n"
     "                       sa_enabled=True, use_centroid_guidance=True, seed=_SEED+2)"),
    ('output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")',
     "output_dir = _OUT"),
]

for old, new in REPLACEMENTS:
    assert old in src, "NOT FOUND: " + old[:70]
    src = src.replace(old, new)

with open(r"E:\Workbuddy AI\Paper Use\_v3_run_seed.py", "w", encoding="utf-8") as f:
    f.write(src)

print("OK: _v3_run_seed.py generated, %d patches applied" % len(REPLACEMENTS))
