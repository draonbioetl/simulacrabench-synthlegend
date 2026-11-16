# SimulacraBench entry of team synthlegend (UNSW Sydney)

This is the code of our entry to **SimulacraBench**, a NeurIPS 2026 competition
(Codabench competition 17822). The Test-phase submission is the single file
`main.py` in this folder: a self-tuning stacked ensemble that predicts, for every
blank `PREDICT` cell of a held-out respondent, a probability vector over the
item's answer options, given the respondent's visible `GIVEN` answers.

| file | SHA-256 |
|---|---|
| `main.py` (version 3.2, the uploaded file) | `a667a3b6f47c0f59dbf31b8131a6a9881c2e2bb88a0ac528dae73d760b733772` |
| upload archive `simulacrabench_submission.zip` (contains only `main.py`) | `c91146ac0164e06550c779883c56f6ecad77b0e3ab92e7f491dd4bbe1f718f94` |

The archive hash depends on file timestamps, so a rebuilt zip has a different
hash. The hash of `main.py` inside it does not change.

The method, the validation and the full configuration are described in our
technical report: "SimulacraBench Technical Report: Self-Tuning Stacking of
Conditional Answer Models", submitted on OpenReview to
NeurIPS.cc/2026/Competition/SimulacraBench.

## What the code does

Everything is fitted at run time, inside the competition container, from the
visible respondents of the instrument being predicted. There is no offline
training, no bundled weights, no external data and no pretrained model.

For each instrument (one `predict(frame, schema)` call), `main.py`:

1. **Encodes the data.** `GIVEN` columns become categorical codes. Columns whose
   levels are numeric bands also get a thermometer encoding. `PREDICT` items
   become K-way targets, with `NA_GATED` last for gated items.
2. **Cross-fits the base models** with 5 folds:
   - smoothed marginals;
   - hierarchical Dirichlet back-off;
   - a batched multinomial logistic regression with interactions;
   - hot-deck donor models, one of them with learned per-column donor weights;
   - multi-task MLPs;
   - a latent-class mixture-of-softmaxes model;
   - time-boxed histogram gradient boosting.
3. **Enforces skip-logic gates** whose parent is a `GIVEN` column, when they hold
   exactly on the visible rows.
4. **Stacks the models per item.** A convex linear pool is fitted by EM, then a
   log-linear pool. Both are fitted on the grader's floored out-of-fold log
   score.
5. **Returns one validated vector per blank cell**, in the grader's canonical
   order.

The code respects the time limit by design:

- It detects the phase budget from the schema's split counts: 900 s in
  Development, 3,600 s in Test.
- It spends 55 % of the budget in Development and 75 % in Test.
- Every model runs against a deadline.
- Any failure falls back to gate-aware marginals, and if those fail, to uniform
  vectors.

## How the competition runs it

The organisers' harness unpacks the zip (with `main.py` at its root) in a
network-isolated container: 1 x H100, 8 CPUs, 16 GB, Python 3.13 with numpy,
pandas, scipy, scikit-learn and torch pre-installed. It imports `main.py` once
and calls `predict(frame, schema)` once per instrument (UNICEF, World Bank,
UNHCR). The wall-clock limit is 900 s (Development) or 3,600 s (Test) for the
whole run.

There is no `requirements.txt` and no `models.txt`. To rebuild the upload:

```bash
zip simulacrabench_submission.zip main.py
```

## Reproducing locally

The competition data are not public, so a local run uses the organisers'
synthetic stand-ins, which have the real shapes. Use the public starter kit at
<https://github.com/SituatedEvals/public> (MIT licence). We used commit
`aa03186`.

```bash
git clone https://github.com/SituatedEvals/public simulacrabench-kit
cd simulacrabench-kit
pip install -r requirements.txt            # the kit's own needs (numpy, pandas, pyarrow, PyYAML)
pip install scipy scikit-learn torch       # what main.py imports (pre-installed on the hosted image)

# synthetic data in the shape of a real instrument
python make_sandbox.py --schema data/unhcr.json --out _sandbox/unhcr

# a directory that holds only main.py
mkdir -p ../sb_submission && cp /path/to/this/folder/main.py ../sb_submission/

# run and score it the way the grader does
# phase 1 = Development layout and 900 s; phase 2 = Test layout and 3600 s
python score.py --submission ../sb_submission --data _sandbox/unhcr \
                --schema data/unhcr.json --phase 2

# check the upload archive against the contract
(cd ../sb_submission && zip ../simulacrabench_submission.zip main.py)
python tools/check_submission_zip.py ../simulacrabench_submission.zip
```

Repeat with `data/unicef.json` and `data/world_bank.json` for the other two
instruments. The scores on sandbox data mean nothing because the answers are
synthetic. Only the organisers can re-run the code on the hidden data.

We tested the code with two setups:

- Python 3.13.15, torch 2.11 (CUDA 12.8), scikit-learn 1.6.1, pandas 2.3.3 and
  numpy 2.1.3, on GPU and on CPU only;
- Python 3.13.16, torch 2.14 (CPU), scikit-learn 1.9.1, pandas 3.0.6 and
  numpy 2.5.4.

A GPU is optional.

## Data and the data-use agreement

This repository contains **no data**: no competition microdata, no sandbox
files and no derived statistics. The SimulacraBench microdata come from UNICEF,
the World Bank and UNHCR. They are not released to participants, and we never
had access to them.

The code runs only inside the organisers' container. It does not write files,
open network connections or print data. It returns probability vectors, nothing
else.

Participation is bound by the competition's data-use terms:

- no attempt to exfiltrate, reconstruct or re-identify records;
- no linking with external datasets;
- no redistribution of microdata.

Anyone re-running this code on the competition data is bound by the same terms.
The hot-deck models compare respondents' `GIVEN` answers only within the frame
the harness passes in (standard hot-deck imputation), and they identify no one.

## Licence and acknowledgement

MIT, see `LICENSE`. The method, the code and the validation were developed with
the help of an AI coding agent (Claude Code, Anthropic).
