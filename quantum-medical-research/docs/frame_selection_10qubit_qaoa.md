# 10-qubit QAOA frame selection for UNet fine-tuning: method and results

Status: **negative result.** A 10-qubit QAOA circuit (classically simulated) selects 100
fine-tuning frames. The circuit finds the optimum of its objective on every split, but the
selected frames do not give a robustly higher Dice than the strongest classical baseline
(`hardest_k`). No quantum advantage is possible or claimed at this size.

All quantum results below come from a **classical simulation** of the circuit (Qiskit
`Statevector` and Aer shot sampling). Nothing was run on quantum hardware.

## Task

A UNet pretrained on Kvasir-SEG is fine-tuned on 100 frames chosen from the 800 Kvasir
training frames (150 steps, lr 3e-5, batch 8). The question is which 100 frames to choose.
The outcome is the Dice gain over the model without fine-tuning, on frames with baseline
Dice < 0.80 ("hard") from held-out CVC-ClinicDB sequences ("CVC rep hard").

CVC sequences are split into a *select* half (may be used to build the objective) and a
*report* half (used only for the final numbers). Five such splits (split seeds 0-4) are
reported; they reuse the same CVC frames in different halves, so they are not independent.

## Code

All in `quantum-medical-research/frame_selection/`:

| file | role |
|---|---|
| `qaoa_block_selection.py` | this experiment: blocks, QUBO, QAOA circuit, fine-tunes, report |
| `learned_qubo_selection.py` | learned surrogate QUBO, frame features, fine-tune/eval helpers |
| `gradient_qubo_selection.py` | data setup, CVC halves, classical baselines |
| `gradient_features.py` | per-frame head-gradient features |
| `frame_selection_qubo.py` | `Qubo` class, greedy / annealing solvers, fine-tune and evaluation |

Extra dependencies: `qiskit` 2.5.2, `qiskit-aer` 0.17.2 (`qiskit-ibm-runtime` only for `--hardware`).
Data, the pretrained checkpoint and all result files are local and not part of this branch.

## Method

### 1. Reduction to 10 binary variables

- **Pool:** the 200 training frames with the highest loss under the pretrained model
  (`hardest_k` is the top 100 of this pool).
- **Blocks:** 10 blocks of 20 frames, equal-size k-means on the 8-dim frame features
  (1 loss, 5 head-gradient PCs, 2 embedding PCs), seed 0.
- **Decision:** choose 5 of the 10 blocks = 100 frames. One qubit per block; 252 feasible
  subsets.

No Dice value is used to build the pool or the blocks.

### 2. Objective (two versions)

**Direction 10, `--objective surrogate`.** The learned surrogate of
`learned_qubo_selection.py`, fitted on fine-tunes of random 100-of-800 subsets, predicts the
CVC select-half hard gain from the mean feature vector `m` of the chosen frames:

    y = b + theta.m + m'Wm - rho (m - mu)' P (m - mu)

Expanded over block indicators this is a QUBO. `rho` (trust region) is the smallest grid
value whose exact optimum has Mahalanobis distance within the 95th percentile of the sampled
subsets.

**Direction 11, `--objective block`.** A 10-variable QUBO fitted directly to measured
outcomes in the same pool: 60 random 5-of-10 block subsets, one fine-tune each, plus 12
`pool_random` runs (72 points per split). `y` = CVC select-half hard gain.
Model `y = b + a.x (+ sum_ij c_ij x_i x_j)`, ridge; linear vs pairwise and
alpha in {1, 10, 100, 1000} chosen by leave-one-out R2. Gate: LOO R2 <= 0 counts as a
failed split.

### 3. Circuit

- QAOA on 10 qubits: `|+>^10`, p = 3 layers of RZ/RZZ cost and RX mixer (depth 130, 270 CX
  before hardware transpilation).
- The cardinality constraint enters as a penalty `A (sum x - 5)^2`, with `A` the smallest
  value (grid 0.1) for which the feasible optimum is the global optimum.
- Angles: COBYLA on the exact statevector expectation, several random starts.
- Sampling: 4096 shots on Aer (ideal, and with a depolarising + readout noise model).
- Selection: the best feasible sampled bitstring.

### 4. Evaluation

- Fine-tune the selected 100 frames with seeds 20-31; compare on the report halves.
- Baselines (same seeds): `hardest_k`, `all_800`, `random_k`, `kcenter`, `craig`,
  `align_topk`, `gradmatch_omp`, `qubo_div`.
- Controls: `pool_random` (5 random blocks of the same pool) and `block_greedy`
  (classical greedy on the same QUBO, fine-tuned where it differs from the circuit's choice).
- Tests: seed-paired t-test with Holm correction over the 8 baselines, **and** a
  frame-bootstrap 95% CI (the stricter test, used for the pass rule).
- Rules were written down before the fine-tunes were run. Pass rule: `qaoa - hardest_k`
  frame-bootstrap CI > 0 on a clear majority of splits (direction 11: at least 4 of 5).

### Commands (from repo root)

    S=quantum-medical-research/frame_selection/qaoa_block_selection.py
    # direction 10
    ./venv/bin/python $S --circuit --split-seed 0
    ./venv/bin/python $S --run     --split-seed 0 --seeds 20 21 22 23 24 25 26 27 28 29 30 31
    ./venv/bin/python $S --report  --split-seed 0
    # direction 11
    ./venv/bin/python $S --sample-blocks 0 60
    ./venv/bin/python $S --circuit --split-seed 0 --objective block
    ./venv/bin/python $S --run     --split-seed 0 --objective block --methods qaoa block_greedy
    ./venv/bin/python $S --report  --split-seed 0 --objective block

`--hardware` submits the saved circuit to an IBM backend (the account must already be saved
with `QiskitRuntimeService.save_account`). This path was only dry-run (transpilation to line
connectivity: p=1 depth 247 / 171 CZ; p=3 depth 566 / 483 CZ) and **never executed on IBM**.

## Results

Gain over no fine-tuning on CVC rep hard, report half, mean of seeds 20-31.

### Circuit quality (simulated)

The best sampled bitstring was the exact optimum (rank 1 of 252) on all five splits for both
objectives.

| objective | P(optimum), exact statevector | uniform over feasible |
|---|---|---|
| surrogate (direction 10) | 0.015 - 0.078 | 0.004 |
| block (direction 11) | 0.057 - 0.356 | 0.004 |

### Direction 10: surrogate objective

| split | qaoa - hardest_k | seed-paired p | frame-bootstrap 95% CI | reading |
|---|---|---|---|---|
| 0 | +0.026 | <0.001 | [+0.007, +0.046] | win |
| 1 | +0.007 | 0.16 | [-0.019, +0.033] | tie |
| 2 | -0.023 | 0.001 | [-0.050, +0.002] | worse |
| 3 | +0.020 | 0.003 | [-0.003, +0.044] | borderline |
| 4 | +0.002 | 0.72 | [-0.021, +0.027] | tie |

Mean +0.006; CI > 0 on 1 of 5 splits. **Pass rule not met.**
Splits (of 5) with frame CI > 0 against each baseline: random_k 5, all_800 4, craig 4,
align_topk 4, kcenter 3, gradmatch_omp 3, qubo_div 2, hardest_k 1.
`block_greedy`, where it differs from the circuit's choice, gave higher Dice than `qaoa` on
splits 1 and 2: the better-optimised subset was not the better subset.

### Direction 11: block objective fitted to measured fine-tunes

| split | model chosen | LOO R2 | greedy = optimum |
|---|---|---|---|
| 0 | linear, alpha 10 | +0.217 | yes |
| 1 | pairwise, alpha 100 | +0.401 | no (rank 2) |
| 2 | linear, alpha 10 | +0.559 | yes |
| 3 | pairwise, alpha 100 | +0.395 | yes |
| 4 | linear, alpha 10 | +0.414 | yes |

| split | qaoa | hardest_k | pool_random | qaoa - hardest_k | seed-paired p | frame-bootstrap 95% CI |
|---|---|---|---|---|---|---|
| 0 | +0.1259 | +0.1158 | +0.1222 | +0.0101 | 0.032 | [-0.0081, +0.0295] |
| 1 | +0.1554 | +0.1497 | +0.1433 | +0.0057 | 0.439 | [-0.0255, +0.0390] |
| 2 | +0.1179 | +0.1206 | +0.1087 | -0.0028 | 0.625 | [-0.0248, +0.0203] |
| 3 | +0.0973 | +0.0972 | +0.0953 | +0.0001 | 0.984 | [-0.0231, +0.0244] |
| 4 | +0.1245 | +0.1013 | +0.0931 | +0.0232 | 0.002 | [-0.0005, +0.0486] |

Mean +0.0073; CI > 0 on 0 of 5 splits. **Pass rule (at least 4 of 5) not met.**
Split 1, `qaoa - block_greedy`: +0.0060, CI [-0.0153, +0.0280] (tie).
`qaoa - pool_random`: CI > 0 on split 4 only.

### Absolute Dice (report halves, mean of seeds 20-31)

The tables above give differences. These are the Dice values themselves: `no_finetune` is the
pretrained model, every other column is after fine-tuning on that method's 100 frames
(`all_800`: all 800 frames). `qaoa D10` / `qaoa D11` are the circuit's selections under the
surrogate and the block objective. Simulated circuit, no QPU.

**CVC rep hard** (primary outcome; frames with pretrained Dice < 0.80)

| split | n | no_finetune | qaoa D10 | qaoa D11 | pool_random | hardest_k | all_800 | random_k | kcenter | craig | align_topk | gradmatch_omp | qubo_div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 115 | 0.4616 | 0.6032 | 0.5875 | 0.5839 | 0.5775 | 0.5755 | 0.5641 | 0.5519 | 0.5339 | 0.5682 | 0.5130 | 0.5652 |
| 1 | 72 | 0.4719 | 0.6285 | 0.6273 | 0.6152 | 0.6216 | 0.5981 | 0.5843 | 0.6067 | 0.5772 | 0.5990 | 0.5755 | 0.6089 |
| 2 | 112 | 0.4984 | 0.5957 | 0.6163 | 0.6071 | 0.6191 | 0.5887 | 0.5740 | 0.6154 | 0.5911 | 0.5946 | 0.5899 | 0.6061 |
| 3 | 101 | 0.4800 | 0.5969 | 0.5773 | 0.5753 | 0.5771 | 0.5667 | 0.5613 | 0.5693 | 0.5369 | 0.5662 | 0.5411 | 0.5691 |
| 4 | 120 | 0.4700 | 0.5735 | 0.5945 | 0.5631 | 0.5713 | 0.5498 | 0.5481 | 0.5457 | 0.5339 | 0.5462 | 0.5362 | 0.5604 |
| mean |  | 0.4764 | 0.5996 | 0.6006 | 0.5889 | 0.5933 | 0.5758 | 0.5664 | 0.5778 | 0.5546 | 0.5748 | 0.5511 | 0.5820 |

Seed standard deviation on this set is 0.009-0.025 per method and split, larger than the
qaoa - hardest_k differences of the means (+0.006 and +0.007).

`block_greedy` (classical greedy on the same QUBO), where it selects a different subset:
D10 split 1 0.6621, split 2 0.6099, split 3 0.5965; D11 split 1 0.6212.

**CVC rep full** (all report-half CVC frames)

| split | n | no_finetune | qaoa D10 | qaoa D11 | pool_random | hardest_k | all_800 | random_k | kcenter | craig | align_topk | gradmatch_omp | qubo_div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 324 | 0.7514 | 0.8048 | 0.7969 | 0.7978 | 0.7921 | 0.7952 | 0.7886 | 0.7847 | 0.7748 | 0.7888 | 0.7577 | 0.7879 |
| 1 | 320 | 0.8175 | 0.8584 | 0.8574 | 0.8549 | 0.8547 | 0.8499 | 0.8448 | 0.8530 | 0.8468 | 0.8496 | 0.8360 | 0.8525 |
| 2 | 324 | 0.7690 | 0.7916 | 0.8037 | 0.8052 | 0.8081 | 0.7993 | 0.7928 | 0.8103 | 0.8016 | 0.7982 | 0.7900 | 0.8043 |
| 3 | 316 | 0.7831 | 0.8193 | 0.8107 | 0.8129 | 0.8110 | 0.8100 | 0.8069 | 0.8105 | 0.7977 | 0.8066 | 0.7897 | 0.8095 |
| 4 | 304 | 0.7369 | 0.7710 | 0.7810 | 0.7707 | 0.7724 | 0.7647 | 0.7636 | 0.7626 | 0.7585 | 0.7600 | 0.7552 | 0.7684 |
| mean |  | 0.7716 | 0.8090 | 0.8099 | 0.8083 | 0.8077 | 0.8038 | 0.7993 | 0.8042 | 0.7959 | 0.8007 | 0.7857 | 0.8045 |

**Kvasir rep hard** (only 13 frames per split; noisy)

| split | n | no_finetune | qaoa D10 | qaoa D11 | pool_random | hardest_k | all_800 | random_k | kcenter | craig | align_topk | gradmatch_omp | qubo_div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 13 | 0.5953 | 0.6145 | 0.6081 | 0.6163 | 0.6194 | 0.6043 | 0.6101 | 0.5868 | 0.6080 | 0.6232 | 0.6249 | 0.6271 |
| 1 | 13 | 0.4554 | 0.5684 | 0.5571 | 0.5481 | 0.5386 | 0.5361 | 0.5288 | 0.5390 | 0.5109 | 0.5483 | 0.5127 | 0.5505 |
| 2 | 13 | 0.5515 | 0.5589 | 0.5538 | 0.5594 | 0.5622 | 0.5480 | 0.5561 | 0.5254 | 0.5421 | 0.5541 | 0.5512 | 0.5683 |
| 3 | 13 | 0.5068 | 0.5556 | 0.5584 | 0.5430 | 0.5381 | 0.5339 | 0.5352 | 0.5238 | 0.5175 | 0.5459 | 0.5431 | 0.5505 |
| 4 | 13 | 0.6786 | 0.7158 | 0.6835 | 0.7062 | 0.7077 | 0.6936 | 0.6950 | 0.6843 | 0.6847 | 0.6949 | 0.6912 | 0.7109 |
| mean |  | 0.5575 | 0.6026 | 0.5922 | 0.5946 | 0.5932 | 0.5832 | 0.5850 | 0.5719 | 0.5727 | 0.5933 | 0.5846 | 0.6015 |

**Kvasir rep full** (in-domain; 100 frames per split)

| split | n | no_finetune | qaoa D10 | qaoa D11 | pool_random | hardest_k | all_800 | random_k | kcenter | craig | align_topk | gradmatch_omp | qubo_div |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 100 | 0.8947 | 0.8948 | 0.8966 | 0.8952 | 0.8958 | 0.8991 | 0.8987 | 0.8980 | 0.8971 | 0.8987 | 0.8934 | 0.8968 |
| 1 | 100 | 0.8743 | 0.8893 | 0.8848 | 0.8870 | 0.8848 | 0.8888 | 0.8866 | 0.8873 | 0.8804 | 0.8879 | 0.8733 | 0.8848 |
| 2 | 100 | 0.8945 | 0.8847 | 0.8809 | 0.8886 | 0.8870 | 0.8898 | 0.8932 | 0.8845 | 0.8892 | 0.8911 | 0.8803 | 0.8872 |
| 3 | 100 | 0.8856 | 0.8838 | 0.8842 | 0.8827 | 0.8822 | 0.8856 | 0.8870 | 0.8839 | 0.8847 | 0.8869 | 0.8784 | 0.8839 |
| 4 | 100 | 0.9076 | 0.9041 | 0.8975 | 0.9030 | 0.8998 | 0.9047 | 0.9073 | 0.9013 | 0.9042 | 0.9023 | 0.8978 | 0.8979 |
| mean |  | 0.8913 | 0.8914 | 0.8888 | 0.8913 | 0.8899 | 0.8936 | 0.8946 | 0.8910 | 0.8911 | 0.8934 | 0.8846 | 0.8901 |

Reading: fine-tuning on any 100-frame selection raises CVC hard Dice from about 0.48 to
0.55-0.60 and CVC full Dice from 0.77 to 0.79-0.81, and leaves in-domain Kvasir Dice
unchanged (0.89). The circuit's selections have the highest mean on CVC (0.600 hard,
0.809-0.810 full), ahead of `hardest_k` by 0.006-0.007 on hard frames and 0.001-0.002 on all
frames; these margins are within noise (see the confidence intervals above). On Kvasir full
they are level with or slightly below the other methods.

## Conclusions

1. The 10-qubit QAOA circuit works as an optimiser in simulation: it samples the exact
   optimum of its QUBO on every split, far above the uniform rate.
2. The selected frames are clearly better than several classical baselines (random_k,
   all_800, craig, and on most splits gradmatch_omp), but only tie with `hardest_k`.
3. Most of the gain comes from restricting to the 200 hardest frames: random blocks from the
   same pool do about as well as the circuit's choice.
4. With a predictive objective (direction 11) the QUBO is linear on 3 of 5 splits and greedy
   reaches the same optimum on 4 of 5, so the circuit adds nothing over sorting by score.

## Limits

- **No quantum advantage:** 252 feasible subsets are enumerated instantly; brute force
  returns the same subset as the circuit.
- **Simulation only:** no QPU run; under hardware noise a p=1 circuit is expected to be close
  to uniform sampling (simulated P(optimum) 0.005 vs 0.004 uniform).
- **Label use:** the objectives use CVC select-half labels; `hardest_k` uses none.
- **Multiple attempts:** these are directions 10 and 11 of a longer series; the report halves
  were viewed for both, so no further variants were tried on them.
