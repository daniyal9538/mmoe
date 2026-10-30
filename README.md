# M-MoE (expert choice): a multi-head mixture-of-experts MLP encoder with learned feature groups

An MLP encoder in which the input features are split into **learned, overlapping groups**. Each of
several **heads** has m small expert MLPs. With **expert-choice routing**, every expert picks the k input
features it reads. A learned **null** lets each head ignore features, and an optional **input budget**
keeps exactly K features in total. Heads differ in granularity (e.g. 4, 16 and 64 experts), so one feature
can sit in a coarse group in one head and a fine group in another. Each expert outputs a few numbers. These
are concatenated and mapped linearly to the embedding z. After training, all gates are **hard 0/1**: the
result is a genuinely sparse network whose groups you can read off.

It was built for self-supervised representation learning on sparse, mostly-missing tabular data (hospital
lab tests), trained as an autoencoder. It works on any [N, F] table with an optional observed-mask.

> **Name clash.** "MMoE" here means *multi-head* MoE over input features. It is **not** the multi-gate MoE
> of Ma et al. (KDD 2018), where task-specific gates mix shared experts per sample. Routing here is
> **static**: it depends on the feature, not the sample.

- **Provenance.** Ported from `Documents/daniyal/mmoemlp` (MIMIC-IV lab autoencoder benchmark, configs
  `mmoe_expert_choice`, `mmoe_ec_cols`, `mmoe_ecs_K*`) on 2026-10-29. Only the expert-choice version is
  kept. The other variants are recorded in [Versions not included](#versions-not-included-record-of-ideas).
- **Tutorial.** [`notebooks/tutorial_mnist.ipynb`](notebooks/tutorial_mnist.ipynb) changes every
  hyperparameter one at a time on MNIST (pixels = features) and shows what it does.

---

## Contents
1. [Quickstart](#quickstart)
2. [Repository layout](#repository-layout)
3. [How the model works](#how-the-model-works)
4. [Hyperparameters](#hyperparameters)
5. [Diagnostics and outputs](#diagnostics-and-outputs)
6. [Results so far](#results-so-far)
7. [Known caveats](#known-caveats)
8. [Versions not included (record of ideas)](#versions-not-included-record-of-ideas)
9. [What is new compared to the original code](#what-is-new-compared-to-the-original-code)
10. [Maintenance](#maintenance)
11. [References](#references)

---

## Quickstart

Plain scripts, no packaging: code lives in `src/mmoe/`, and scripts and notebooks put `src/` on `sys.path`.

```bash
python -m venv .venv && .venv\Scripts\activate        # or reuse an env with torch (CUDA build recommended)
pip install -r requirements.txt

python -m pytest tests                                # 18 unit tests, ~10 s
python scripts/train.py --config configs/mnist.yaml --dataset mnist --out runs/mnist_baseline
python scripts/train.py --config configs/default.yaml --data my_table.npz --out runs/x --set model.r=4 train.seed=1

python notebooks/build_tutorial_mnist.py              # regenerate the notebook from its builder
jupyter nbconvert --to notebook --execute --inplace notebooks/tutorial_mnist.ipynb --ExecutePreprocessor.timeout=3600
```

From Python:

```python
import sys; sys.path.insert(0, "src")
from mmoe import fit, model_config, train_config, encode, assignments

mcfg = model_config(experts_per_head=[4, 16, 64], r=2, null_mode="per_head")
tcfg = train_config(epochs=100)
res = fit(mcfg, tcfg, {"train": (X_tr, O_tr), "val": (X_va, O_va)})   # O = 0/1 observed mask or None
z = encode(res.model, X_te, O_te)                                     # [N, latent_dim]
groups = assignments(res.model.encoder)                               # per head: G [U, m], keep [U]
```

`--data` expects an `.npz` with `x_train`, `x_val` and optionally `obs_train`, `obs_val`. Values should
be scaled (e.g. robust z-scores). Values at unobserved entries are ignored by the loss but still enter the
encoder, so set them to 0.

## Repository layout

```
configs/default.yaml        every hyperparameter with its default (= the MIMIC-IV mmoe_expert_choice run)
configs/mnist.yaml          tutorial overrides (no obs flags, 20-epoch schedule)
src/mmoe/config.py          ModelConfig / PhaseConfig / TrainConfig dataclasses, YAML loading, validation
src/mmoe/gates.py           ExpertChoiceGate, HardConcreteGate, FixedNullGate, TopKGate, ConcreteSelectorGate, initialisers
src/mmoe/experts.py         ExpertBank: m masked 2-layer MLPs per head, vectorised
src/mmoe/input_gate.py      global feature budget + value/flag column gates
src/mmoe/encoder.py         MMoEEncoder (heads, null, budget, hardening, regularizers, effective params)
src/mmoe/schedule.py        PhaseController: warm-up -> gate learning -> hardened fine-tuning
src/mmoe/model.py           Decoder, reconstruction loss, AutoEncoder, optimiser groups, LR schedule
src/mmoe/prior.py           |corr| prior over co-observed pairs (for gate_init=prior_corr)
src/mmoe/diagnostics.py     per-head gate diagnostics, hard assignments
src/mmoe/train.py           fit(), encode(), evaluate_recon()
src/mmoe/data.py            MNIST loader (no torchvision), occlusion for missing-data demos
scripts/train.py            CLI: train one config, save history / assignments / weights
notebooks/                  tutorial_mnist.ipynb + its builder script
tests/test_mmoe.py          unit tests
```

## How the model works

```
x_val [B,F] ─┐
             ├─ concat ─► x [B, R]      R = 2F with observed flags, F without
x_obs [B,F] ─┘     (optional input budget: features outside the K selected are closed everywhere)

for each head h with m_h experts:
    keep_h [U]           null gate: which units the head may use        (per_head | global | off)
    logits_h [U, m_h]    routing scores
    g_h = top-k per expert column over eligible units (keep_h > 0)      k = ceil(r * U / m_h)
    G_h = g_h * keep_h
    x @ (W1_h ⊙ blockmask(G_h)) -> GELU -> dropout -> per-expert u -> d_e   => [B, m_h * d_e]

concat over heads -> Linear -> z [B, d_z] -> decoder -> (values [F], observed logits [F])
```

- **Units.** By default a unit is a feature (U = F): its value and its was-observed flag share every gate.
  With `column_units`, value and flag are separate units (U = 2F).
- **Expert choice over features.** Each expert (a column of `logits_h`) keeps its k highest-scoring
  eligible units. Every expert has exactly k units, so load balance is exact and no expert can die.
  The flip side: a unit can be picked by many experts or by none (an *orphan*). Zhou et al. (2022) route
  tokens to experts per sample. Here the experts choose features, once, for all samples.
- **Straight-through learning.** The forward pass uses the hard top-k mask. Gradients reach the logits
  through a surrogate (`ec_surrogate`): sigmoid(logits), or k·softmax over units per expert.
- **Null.** Hard-concrete keep gates (Louizos et al., 2018) with an L0 price, or a fixed-size null. Null
  units are ineligible, so experts spend their seats on kept units.
- **Input budget.** Optional global gate that keeps exactly K features, using K concrete selector units
  (Abid et al., 2019) or straight-through top-K. Capacity then becomes k = ⌈r·K/m⌉. With
  `split_value_flag`, each kept feature's value and flag columns get their own L0-priced gates.
- **Objective.** Autoencoder: Huber on observed values (optionally re-weighted per feature) +
  `obs_bce_weight`·BCE on the observed flags. The decoder always reconstructs **all** F features, so
  dropping a feature never makes the task easier.
- **Regularizers.** `L = L_recon + λ_mem·L_mem + λ_null·L_null (+ λ_col·L_col)`, weights ramped by the
  schedule. There is no load-balance loss (balance is exact).

### Training phases (`schedule.py`)

| phase | epochs (default) | gates | optimiser |
|---|---|---|---|
| 0 dense warm-up | [0, `warmup_end`=10) | all forced to 1: a dense MLP | no regularizers |
| 1 gate learning | [10, `harden_start`=60) | stochastic (hard-concrete samples, straight-through top-k); temperature `temp_start` → `temp_end` | λ ramp to target by `lambda_ramp_end`=35; gate logits at `gate_lr_mult`× LR, no weight decay |
| 2 hardened fine-tune | [60, end) | fixed 0/1; logits frozen | weights at `finetune_lr_mult`× LR; early stopping |

Hardening order: input budget → keep gates → routing (experts choose among surviving units). **Only
hardened epochs can be selected as the checkpoint** (lowest val reconstruction, regularizers excluded), so
the returned model is always the real sparse model.

## Hyperparameters

All keys live in [`configs/default.yaml`](configs/default.yaml) / [`src/mmoe/config.py`](src/mmoe/config.py).
Defaults are those of the MIMIC-IV `mmoe_expert_choice` run unless marked *new*. In the YAML, write
`null_mode: "off"` with quotes, because bare `off` is boolean false (validation catches this).

### Architecture

| key | default | values | controls | affects / expected effect |
|---|---|---|---|---|
| `experts_per_head` | `[4, 16, 64]` | list of ints | number of heads and each head's expert count (granularity) | Coarse heads give broad groups (organ-system panels, large image regions); fine heads give tight co-varying sets. On MIMIC, `[4]` alone was far worse, `[64]` alone slightly worse than multi-head. Also sets the readout width Σ m·d_e. |
| `expert_hidden` (u) | 16 | int | hidden width of each expert | capacity per group; dense W1 size R × (m·u) per head |
| `expert_out` (d_e) | 2 | int | numbers each expert passes on | per-group bottleneck. d_e = 1 is easier to interpret but loses information. |
| `latent_dim` (d_z) | 64 | int | embedding size | probe quality, decoder input |
| `dropout` | 0.1 | float | dropout on expert hidden layers | regularisation |
| `use_obs_flags` | true | bool | input = [value, was-observed flag] per feature; false = values only | lets the model tell "missing" from "zero" and use measurement patterns. Set false for fully observed data (MNIST). *new: false option* |
| `column_units` | false | bool | value and flag are separate routing units (U = 2F) | a head can use "was it measured" without the value, or the reverse. Doubles gate count. On MIMIC: best `expired` AUROC of the M-MoE variants. |

### Expert-choice routing

| key | default | values | controls | affects / expected effect |
|---|---|---|---|---|
| `r` | 2 | float > 0 | capacity factor, k = ⌈r·U/m⌉ (U → K with a budget): the *average* number of experts per unit | overlap (↑r gives more shared units), orphans (↓r gives more), effective parameters (∝ r), closeness to a dense MLP |
| `capacity` | null | int or list per head | fixed k per expert, overriding r | decouples group size from m and U, e.g. to keep group size constant across heads. *new* |
| `ec_surrogate` | sigmoid | sigmoid, softmax | gradient path of the straight-through top-k | sigmoid: independent per (unit, expert) signals. softmax: units compete for seats, so routing changes more decisively. *new: softmax* |
| `ec_noise` | 0.0 | float ≥ 0 | Gumbel noise × temperature on scores before top-k (training only) | exploration near the selection boundary; anneals with temperature. 0 reproduces the original runs. *new* |

### Null (keep) gates

| key | default | values | controls | affects / expected effect |
|---|---|---|---|---|
| `null_mode` | per_head | per_head, global, off | own keep gate per head / one shared / none | per-head lets heads specialise (a fine head drops what a coarse head keeps); global gives one consistent selected set; off is the upper bound on inputs (on MIMIC all three were within noise) |
| `null_type` | learned | learned, fixed | hard-concrete + L0 price vs drop exactly `null_q` | fixed is a control for null *size*: does choosing how many to drop matter? |
| `null_q` | 0.25 | [0, 1) | fraction dropped by the fixed null | — |
| `keep_init` | 3.0 | float | initial keep logit (P ≈ 0.95) | lower values start nearer the boundary, so the null grows sooner |
| `lambda_null` | 1e-3 | float ≥ 0 | L0 price on keeping a unit | number of kept units; rare, noisy or redundant features are dropped first |

### Initialisation

| key | default | values | controls | affects / expected effect |
|---|---|---|---|---|
| `gate_init` | prior_corr | prior_corr, random | routing start: spectral clustering of train \|corr\| (k = m per head) vs ⌈r⌉ random experts per unit | prior gives coherent starting groups and is expected to be more stable across seeds; random tests whether the grouping comes from data or from the prior |
| `init_scale` | 2.0 | float | ±scale on initial routing logits | stickiness of the start: larger values mean less movement for a given gate LR |
| `init_noise` | 0.1 | float | N(0, σ) jitter on routing logits | breaks ties; seed diversity |
| `init_fill` | binary | binary, graded | `prior_corr` only: binary gives ±`init_scale` by cluster; graded ranks every unit by its mean affinity to each cluster | with binary, seats beyond the cluster size (≈ half of them at r = 2, more if clusters are uneven) and seats freed by the null go to whichever unit the jitter favours. Graded fills them with the units most similar to the cluster. Any [U, U] affinity can be passed as the prior (e.g. a pixel-distance kernel for a spatial prior). *new* |
| `prior_min_coobs` | 50 | int | pairs co-observed fewer times get \|corr\| = 0 | prior reliability for rarely measured features |
| `train_gates` | true | bool | false freezes the initial assignment (hardened at build, warm-up still applies) | control: `prior_frozen` asks whether learning improves on the prior; random + frozen asks whether *which* features group matters |

### Global input budget

| key | default | values | controls | affects / expected effect |
|---|---|---|---|---|
| `input_budget` (K) | null | int or null | keep exactly K features in front of all heads; expert capacity becomes ⌈r·K/m⌉ | feature selection inside the model; recon of dropped features must come from the others |
| `budget_type` | concrete | concrete, topk | K concrete selector units vs straight-through top-K | concrete: every feature gets a gradient, and the selection learns. topk: the selection stays frozen at init (see caveats). (Original default was topk.) |
| `split_value_flag` | false | bool | separate L0 gates for each kept feature's value and flag column | which features are used through their value, flag or both |
| `lambda_col` | 1e-3 | float | L0 price on open value/flag columns | pushes kept features to value-only or flag-only |

### Regularizers

| key | default | controls | affects |
|---|---|---|---|
| `lambda_mem` | 1e-2 | membership cap ReLU(Σ_j sigmoid(logit) − r)² | under expert choice this is only a weak shrinkage of routing logits; membership is decided by the ranking. Kept for parity with the original runs. |
| `lambda_null` | 1e-3 | see Null | |
| `lambda_col` | 1e-3 | see Input budget | |

### Gate schedule (`phases`)

| key | default | controls | affects |
|---|---|---|---|
| `warmup_end` | 10 | epochs of dense training first | quality of the expert weights when routing starts; with 0, routing is decided by random weights |
| `harden_start` | 60 | first hardened epoch | exploration time vs fine-tuning time; checkpoints come only from ≥ this epoch |
| `lambda_ramp_end` | 35 | regularizer ramp end | early pressure on the null |
| `temp_start` / `temp_end` | 1.0 / 0.1 | hard-concrete temperature and `ec_noise` scale | stochasticity of keep / column / budget gates. Routing is a deterministic top-k, so temperature only matters for it through `ec_noise`. |
| `gate_lr_mult` | 10 | gate-logit LR multiplier in phase 1 | **how far routing moves** from its initial assignment |
| `finetune_lr_mult` | 0.3 | weight LR multiplier after hardening | final fit |

### Training (`train:`)

| key | default | controls |
|---|---|---|
| `epochs`, `batch_size`, `lr`, `betas`, `weight_decay` | 100, 512, 1e-3, (0.9, 0.999), 1e-4 | AdamW; no weight decay on biases or gate logits |
| `warmup_epochs` | 5 | linear LR warm-up, then cosine to 0 |
| `grad_clip` | 1.0 | gradient-norm clipping |
| `decoder_hidden`, `decoder_dropout` | [256, 512], 0.1 | shared decoder z → 256 → 512 → outputs |
| `huber_delta` | 1.0 | Huber on observed values |
| `obs_bce_weight` | 0.2 | weight of observed-flag BCE |
| `feature_loss_floor` | null | per-feature weight 1/max(coverage, floor). Rare features count equally above the floor. Used for training and checkpoint selection. |
| `patience` | 10 | early stopping over hardened epochs |
| `bf16` | true | autocast on CUDA (loss in fp32) |
| `seed` | 0 | everything: weights, clustering, jitter, data order |

## Diagnostics and outputs

`fit()` returns `FitResult(model, history, best_epoch, best_val, params)`. `history` has one row per epoch,
with losses, phase, temperature and per-head columns `h{i}_*` (from `diagnostics.py`, computed on the hard
assignments):

| column | meaning |
|---|---|
| `k`, `sizes` | seats per expert, units per expert |
| `n_kept`, `n_null` | units the head uses / ignores |
| `mean_memberships` | mean number of experts reading a kept unit (overlap) |
| `n_orphan` | kept units no expert picked (input but unused) *new* |
| `changed_vs_init` | share of **all** (unit, expert) pairs whose membership differs from the initial assignment (gained or lost) *new*. Not the share of memberships that moved: multiply by ≈ U / (2k) for that (MNIST baseline: 0.25 / 0.12 / 0.036 for heads 0 / 1 / 2 ≈ 25% / 50% / 55% of memberships) |
| `entropy_keep` | indecision of the keep gates (→ 0) |
| `routing_margin` | mean gap between each expert's k-th and (k+1)-th logit (how settled the top-k is) *new* |
| `in_kept`, `in_both`, `in_value_only`, `in_flag_only` | input budget summary |

`params` has dense and **effective** encoder parameter counts. Effective counts exclude masked W1 entries:
W1 is stored densely and masked, so the optimiser holds the dense count.

## Results so far

### MIMIC-IV labs (original repo, `Documents/daniyal/mmoemlp/results/summary.csv`)
F = 372 lab features, 319k train admissions, 92% missing. Autoencoder pre-training, then linear probes on
the test split. Mean ± sd over seeds (n). These numbers come from the original code; this repo's
expert-choice path is a port of it.

| model | recon Huber ↓ | `expired` AUROC | core disease groups macro AUROC | cardiac subgroups macro AUPRC | n |
|---|---|---|---|---|---|
| mlp_ae (dense MLP AE) | 0.0096 ± 0.0001 | 0.878 ± 0.003 | 0.712 ± 0.001 | 0.342 ± 0.004 | 5 |
| mlp_ae_wide (param-matched) | 0.0084 ± 0.0001 | 0.881 ± 0.001 | 0.714 ± 0.001 | 0.341 ± 0.004 | 5 |
| mmoe (feature choice, default) | 0.0092 ± 0.0001 | 0.885 ± 0.002 | 0.712 ± 0.001 | 0.352 ± 0.003 | 5 |
| **mmoe_expert_choice** | 0.0086 ± 0.0003 | 0.885 ± 0.002 | 0.713 ± 0.001 | 0.354 ± 0.001 | 3 |
| **mmoe_ec_cols** (`column_units`) | 0.0088 ± 0.0001 | **0.889 ± 0.001** | **0.715 ± 0.000** | 0.354 ± 0.003 | 3 |
| mmoe_ec_cols, λ_null × 10 | 0.0088 ± 0.0002 | 0.888 ± 0.000 | 0.715 ± 0.000 | 0.353 ± 0.004 | 3 |
| EC + concrete budget K = 186 | 0.0290 ± 0.0003 | 0.879 ± 0.004 | 0.708 ± 0.001 | 0.340 ± 0.001 | 3 |
| EC + concrete budget K = 93 | 0.060 ± 0.010 | 0.877 ± 0.002 | 0.706 ± 0.002 | 0.346 ± 0.007 | 3 |
| EC + concrete budget K = 47 | 0.106 ± 0.009 | 0.870 ± 0.005 | 0.701 ± 0.005 | 0.358 ± 0.005 | 3 |
| EC + concrete budget K = 25 | 0.163 ± 0.019 | 0.857 ± 0.007 | 0.691 ± 0.005 | 0.330 ± 0.016 | 3 |
| EC + top-K budget K = 186 | 0.221 ± 0.016 | 0.864 ± 0.009 | 0.681 ± 0.003 | 0.311 ± 0.007 | 3 |
| EC + top-K budget K = 25 | 0.296 ± 0.024 | 0.808 ± 0.022 | 0.639 ± 0.028 | 0.250 ± 0.009 | 3 |
| ft_transformer (reference) | 0.0122 ± 0.0007 | 0.890 ± 0.002 | 0.717 ± 0.001 | 0.371 ± 0.004 | 5 |
| raw features + probe (reference) | — | 0.895 | 0.736 | 0.409 | 1 |

Takeaways:
1. Expert choice matches or beats feature choice on every probe, with better reconstruction and fewer
   moving parts (no load-balance loss, no dead-expert revival).
2. Routing value and flag separately (`column_units`) gives the best M-MoE probes.
3. The concrete budget degrades gracefully down to K ≈ 47 labs (an 8× input reduction). The top-K budget
   does not learn its selection and is clearly worse at every K.
4. Raw features and the FT-Transformer are still ahead on probes, so the M-MoE's value is sparsity and
   interpretable groups at near-dense quality, not state-of-the-art probes.

### MNIST tutorial (`notebooks/tutorial_mnist.ipynb`, 1 seed, 20 epochs, 784 pixels)
Baseline: test recon 0.0055, linear-probe digit accuracy 0.912, 91k effective vs 1.14M dense encoder
parameters. One hyperparameter changed at a time:

| knob | finding |
|---|---|
| `experts_per_head` | multi-head beats every single head on recon and probe. Single head: finer is better (`[4]` 0.83, `[16]` 0.89, `[64]` 0.91 probe). In the baseline, the 64-expert head carries the digit information (per-head probe 0.91 vs 0.50 for the 4-expert head). |
| `expert_hidden`/`expert_out` | pure capacity knob: recon improves with size, probe flat; d_e = 1 is nearly free |
| `r` | monotone trade-off: ↑r gives better recon, more overlap and fewer orphans (head 2: 51 → 21 → 5 for r = 1, 2, 4), with parameters ∝ r |
| `lambda_null` | sets null size (784 → ~490 → ~310 kept pixels for λ = 0, 1e-3, 1e-2), mostly border, at no recon/probe cost |
| `null_mode`/`null_type` | probes within ±0.001; null off reconstructs best but leaves ~146 orphans in the fine head |
| `train_gates`/`gate_init` | **negative result**: frozen routing (prior or random) reconstructs slightly *better* than learned routing, with probes ~0.002 lower. Learned routing gave no clear gain on MNIST (it did on MIMIC). Seed stability ARI 0.46–0.75, and the prior does not help it. |
| `phases.gate_lr_mult` | main lever on how far gates move: at ×1 the null never forms and head 0's routing barely moves (5% of memberships vs 25%; heads 1–2 ≈ 39% / 24% vs ≈ 49% / 56%); ×10 vs ×30 is similar. No warm-up did not hurt. |
| `ec_surrogate`/`ec_noise`/`lambda_mem` | softmax ≈ sigmoid; `ec_noise = 1` hurts recon by ~25%; `lambda_mem = 0.1` cuts orphans 21 → 3 with no metric change |
| `input_budget`/`budget_type` | concrete learns a digit-centred selection; top-K stays at its random start (border pixels included): lower recon error but clearly worse probe (0.865 vs 0.889 at K = 150) |
| `use_obs_flags`/`column_units` (occluded MNIST) | probe 0.821 (values only), 0.833 (flags), 0.842 (separate value/flag units): same ordering as on MIMIC |

The \|corr\| prior gives **scattered**, not spatially compact, pixel groups: MNIST pixel correlation
follows digit strokes, not adjacency.

## Known caveats

- **Top-K budget does not learn.** A feature outside the top K is ineligible for every expert, so its
  budget score gets no gradient and the random initial selection freezes. Use `budget_type: concrete`.
  `topk` is kept for reproducibility.
- **Orphans.** Expert choice guarantees expert balance, not unit coverage: a kept unit can be picked by no
  expert. Watch `n_orphan`. Raise `r`, or use `null_mode` so the keep gate and routing agree.
- **`lambda_mem` is weak under expert choice.** There is no well-defined expected membership, so it acts
  on sigmoid(logits) and only shrinks them.
- **Temperature does not touch routing** unless `ec_noise > 0`: the expert-choice top-k is deterministic
  given the logits.
- **Dense storage.** W1 is [R, m·u] per head and masked, so memory and FLOPs are those of the dense layer
  (R × Σ m·u). Sparsity is in the *effective* parameter count only.
- **Concrete budget and capacity.** While the concrete selection is soft, experts use full capacity
  ⌈r·F/m⌉ so every feature stays reachable. Capacity drops to ⌈r·K/m⌉ at hardening, so phase 2 must re-fit.
- **Static routing.** Groups are the same for every sample. Sample-dependent routing would be a different
  model (see below).
- **Learned vs frozen routing is data-dependent.** On MNIST the frozen controls matched or beat learned
  routing. Always run `train_gates: false` (prior and random) as controls before claiming the learned
  grouping matters.
- **Recon can reward the wrong selection.** The frozen random top-K budget beat the learned concrete one
  on MNIST reconstruction but lost on the probe. Report a downstream probe, not only recon.
- **Membership exceeds r when the null is active.** k is computed from all U units, so dropping units
  concentrates the same seats on fewer units. Use `capacity` to fix group size independently.
- **Seeds.** `use_deterministic_algorithms` is not forced, so GPU runs can differ slightly run to run.

## Versions not included (record of ideas)

These variants existed in `Documents/daniyal/mmoemlp` (with results above or in its `results/`). They
were deliberately left out of this repo, which keeps only the expert-choice path.

| version | what it was | config flag(s) there | why it is not here / what we learned |
|---|---|---|---|
| **Feature-choice routing** | each (feature, expert) pair has an independent hard-concrete membership gate. Features choose experts. | `routing: feature_choice` (was the default) | Needed a load-balance loss (CV² of expected expert sizes, Shazeer et al. 2017) and dead-expert revival (every 5 epochs, move half of the largest expert's features into an expert with < 1 expected feature, re-init its weights, zero its Adam moments). EC matched or beat it without either. |
| **Membership `soft_cap`** | feature choice + L_mem penalising more than r expected memberships | `membership: soft_cap` | Only meaningful with feature choice. |
| **Membership `partition`** | straight-through Gumbel-softmax: exactly one expert per kept feature | `membership: partition` | Crisp modules, perfect for module-level NMI. Slightly worse recon (0.0100) and probes. |
| **Membership `free`** | independent gates, no cap | `membership: free` | Drifts toward a dense MLP; recon 0.0087, probes on par. Groups less distinct. |
| **Matched random control** | random memberships with the same per-head membership and null counts as a reference (Stage A) run, frozen | `mmoe_random_fixed` (`match_stage_a: true`) | Needs a reference run's `modules.csv`. Probes were close to learned (cardiac AUPRC 0.354), which suggests that on MIMIC much of the benefit comes from sparse multi-granular structure rather than exactly *which* labs group. Worth re-testing with EC. |
| **gated_ft** | FT-Transformer where each layer-1 attention head has a learned per-lab gate (same hard-concrete gates and phases). Gates act on attention keys (log g added to logits), layers 2–3 see a lab if any head reads it. Unread labs are cropped from the sequence after hardening. Exploration floor p_floor = 0.05 so closed labs can reopen. `gate_init` random / prior_corr / open; `sparsity` l0 / matched_budget (K from the M-MoE run). | `configs/models/gated_ft.yaml`, `docs/gated_ft.md` | A different backbone. Probes ≈ FT-Transformer (expired 0.890) with selection exact through the whole model, but grouping exact only in layer 1. |
| **Benchmark harness** | 5-seed Stage A/B/C protocol, param-matched `mlp_ae_wide`, baselines (feature_gnn, GRAPE, FT-Transformer, TabNet, SAINT, PCA-64, raw), linear probes on diagnosis groups, per-head probes, module NMI vs lab category/fluid, module stability (ARI across seeds), paired tests vs mlp_ae | `src/evaluate.py`, `src/aggregate.py`, `scripts/` | Dataset-specific (MIMIC-IV cohort, ICD-derived labels). This repo is the model only. |
| **Expert-choice budget with a straight-through top-K** | first lab-budget sweep | `mmoe_ecb_K*` | **Kept** as `budget_type: topk` for reproducibility, but it is a negative result (see caveats). |

## What is new compared to the original code

The expert-choice path is ported faithfully: same gates, initialisers, phases, defaults and hardening
order. These additions are new and **off by default**, except the budget default, which changed to
`concrete`:

- `use_obs_flags: false` (value-only input, for fully observed data), with generic feature naming (no "labs").
- `capacity` (fixed k per expert, int or per head) and float `r`.
- `ec_surrogate: softmax` and `ec_noise` (exploration).
- Diagnostics: `n_orphan`, `changed_vs_init`, `routing_margin`; the `init_mask{h}` buffers record the
  initial routing.
- Removed: dead-expert revival (a no-op under EC), the load-balance loss (EC is exactly balanced),
  partition/free membership, Stage-A matching.
- `fit()` keeps the best hardened checkpoint in memory and returns it; the CLI writes it to disk.

## Maintenance

- **Tests:** `python -m pytest tests` must pass before every commit. They cover exact EC capacity, null
  ineligibility, closed or unrouted units having zero influence, straight-through gradients, hardening
  freeze, budget exactness, value/flag orphan repair, config validation, the phase schedule, the prior
  and a CPU training smoke test.
- **Adding a hyperparameter:** (1) field + comment in `config.py` (and `validate()` if constrained),
  (2) `configs/default.yaml`, (3) the tables above (default, values, controls, affects), (4) a test,
  (5) a tutorial cell if it is research-relevant, (6) `CHANGELOG.md`.
- **Notebook:** edit `notebooks/build_tutorial_mnist.py`, not the `.ipynb`. Regenerate and re-execute.
- **Versioning:** `src/mmoe/__init__.py::__version__` + `CHANGELOG.md` (semver; bump minor for new
  hyperparameters, major for changed defaults).
- **Not tracked:** `data/` (MNIST cache), `runs/`, `*.pt`.

## References

- Zhou et al., *Mixture-of-Experts with Expert Choice Routing*, NeurIPS 2022.
- Louizos, Welling, Kingma, *Learning Sparse Neural Networks through L0 Regularization*, ICLR 2018.
- Abid, Balin, Zou, *Concrete Autoencoders for Differentiable Feature Selection and Reconstruction*, ICML 2019.
- Shazeer et al., *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*, ICLR 2017.
- Ma et al., *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018 (the *other* MMoE).
- Jang, Gu, Poole, *Categorical Reparameterization with Gumbel-Softmax*, ICLR 2017.
