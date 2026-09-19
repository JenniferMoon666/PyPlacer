# PyPlacer — Adaptive Scale-Aware Post-Processing Refinement for FPGA Placement

Post-placement refinement framework built on top of
[DREAMPlaceFPGA](https://github.com/limitmhw/DREAMPlaceFPGA).
It takes DREAMPlaceFPGA's legalized output as the initial solution and improves
weighted HPWL by an adaptive, scale-dependent strategy.

**Result: 25.3% HPWL reduction on ISPD 2016 FPGA-example1 over the DREAMPlaceFPGA
baseline** (mean over five random seeds), with all outputs verified fully legal.

---

## Key results

All numbers are measured with the **same weighted-HPWL metric** ($w_x = 0.7$,
$w_y = 1.2$) applied to both the baseline and our output. The baseline is obtained
by loading DREAMPlaceFPGA's own final placement and recomputing HPWL with the
identical code path — **not** by quoting published figures.

| Benchmark | Instances | Movable | DREAMPlaceFPGA | PyPlacer | Improvement | Runtime |
|---|---:|---:|---:|---:|---:|---:|
| FPGA-example1 | 3,336 | 3,264 | 10,978 | **8,201 $\pm$ 53** | **25.3%** | 2,613 s |
| FPGA-example2 | 542,239 | 541,783 | 2,891,948 | **2,723,751** | 5.8% | 3,427 s |
| FPGA-example3 | 427,800 | 427,194 | 7,766,552 | **7,658,226** | 1.4% | *(pending)* |
| FPGA-example4 | 844,184 | 843,578 | 8,174,005 | **7,852,883** | 3.9% | *(pending)* |

### Why the baseline is re-measured

Directional weights matter. On FPGA-example1, a commonly cited baseline figure is
13,562; re-measuring DREAMPlaceFPGA's own output with the same metric gives
**10,978** — a 19% difference. Quoting the published number would inflate the
apparent gain from 25.3% to 39.5%. We therefore recompute the baseline in-repo.

### Quality–runtime trade-off

Refinement is strongly front-loaded. On FPGA-example1:

| Time (s) | 0 | 151 | 317 | 654 | 1374 | 2286 | 2613 |
|---|---:|---:|---:|---:|---:|---:|---:|
| HPWL | 10,978 | 10,523 | 8,419 | 8,381 | 8,248 | 8,210 | 8,209 |

**92% of the total gain is obtained within the first 12% of the runtime.**

---

## Method

```
DREAMPlaceFPGA (GP + Legalization)
              │
              ▼
        size: N_m < 50K ?
         ╱            ╲
    Stage A          Stage B
  (small circuit)  (large circuit)
  multi-round SA   GPU-batched DP
  + window DP      (PyTorch scatter_reduce)
  T0: 2.0→1.0→0.5  DSP/BRAM/IO swap
```

- **Stage A** ($N_m <$ 50K): three simulated-annealing rounds with descending
  initial temperatures, each followed by window-based detailed placement.
  Full-instance swaps are allowed because the search space stays tractable.
- **Stage B** ($N_m \ge$ 50K): `GPUBatchDP` evaluates $10^4$–$10^5$ candidate
  moves and swaps in a single GPU pass using `torch.scatter_reduce`, then commits
  the most profitable 1–5%. 18–30$\times$ faster than the CPU implementation.

---

## Requirements

- Python 3.11+
- PyTorch 2.8.0+ (CUDA recommended)
- NumPy
- DREAMPlaceFPGA, for GP+LG initial solutions
- ISPD 2016 FPGA placement benchmarks

---

## Quick start

```bash
# 1. Produce (or reuse) DREAMPlaceFPGA's legalized placement
#    Outputs: DREAMPlaceFPGA-local/results/FPGA-example1.final.pl

# 2. Run refinement (defaults: seed=42, full time budget)
export CUDA_VISIBLE_DEVICES=0
python main_v3_freeze.py --dpfpga FPGA-example1

# Run all four benchmarks
python main_v3_freeze.py --dpfpga
```

### Reproducing the multi-seed experiment

The released runner exposes seed and time-budget control via environment
variables (the original release hard-coded both):

```bash
export SEED=42         # WindowDP rounds use SEED, SEED+1, SEED+2
export TIME_SCALE=1.0  # scales the 1200/600/300 s (small) or 5400/1800/900 s (large) budgets
export OUT_DIR=./out/seed_42
python _v3_run_seed.py --dpfpga FPGA-example1
```

Five seeds (42–46) on FPGA-example1 give
8,233 / 8,177 / 8,169 / 8,279 / 8,149 → **8,201 $\pm$ 53**.

---

## Legality

Every refined placement is verified exhaustively:

| Check | example1 | example2 | example3 | example4 |
|---|:--:|:--:|:--:|:--:|
| Out-of-bound | 0 | 0 | 0 | 0 |
| Site-type mismatch | 0 | 0 | 0 | 0 |
| Sites over capacity | 0 | 0 | 0 | 0 |

The DREAMPlaceFPGA inputs pass the same check, confirming the checker is not
vacuously permissive.

---

## Repository layout

```
benchmark.py          ISPD 2016 benchmark parser + weighted HPWL
global_placement.py   ePlace-style global placement (legacy path)
legalization.py       Legalization routines
main_v39.py           Core modules: WindowDP, GPUBatchDP, GPUHPWLTracker
main_v3_freeze.py     Main entry point (V3 freeze)
main_v41.py           GPUBatchDP implementation
_v3_run_seed.py       Parameterised runner (seed / time-scale / output dir)
```

---

## Citation

```bibtex
@misc{li2026pyplacer,
  title  = {Adaptive Scale-Aware Post-Processing Refinement for FPGA Placement},
  author = {Li, Yanze},
  year   = {2026},
  note   = {Preprint}
}
```

---

## License

MIT License.

## Acknowledgements

Built on [DREAMPlaceFPGA](https://github.com/limitmhw/DREAMPlaceFPGA) and evaluated
on the [ISPD 2016 FPGA Placement Contest](https://www.ispd.cc/contests/16/)
benchmarks.
