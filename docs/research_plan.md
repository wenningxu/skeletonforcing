# Joint-Time Valley Forcing — research contract

> HISTORICAL v1. Superseded by reports/approval_report_v2_zh.md and
> configs/overfit_v2.json: redundant 266D motion + official UMT5, overfit first.
> Do not execute the obsolete XYZ/GloVe plan below.

Updated 2026-09-23. Total authorized project budget: USD 500, including storage.

## Hypothesis and representation

Train a new, encoder-free flow model on canonical-world XYZ positions (T,22,3),
in metres, at 20 Hz. Recover XYZ deterministically from the publisher's HumanML3D
263D features. No VAE, latent bottleneck, pretrained motion encoder, or decoder
is part of generation. Each joint/frame has its own noise level. Anchor XYZ
coordinates have sigma=0 during training, initialization and every sampling step.
Use torch.where after each update, including the final output, to preserve them.
This guarantees coordinates, NOT feasible anatomy, physical correctness or a
correct conditional distribution. Inconsistent input constraints remain possible.

Let d(t,j) be the nearest anchor distance in time plus skeleton graph distance.
For progress p in [0,1], a=0.75*d/(1+d), sigma=1-clamp((p-a)/(1-a),0,1).
Unknown coordinates start at sigma=1 and end at sigma=0; near anchors denoise
earlier. Anchors are excluded from this formula and remain sigma=0.
Training selects a complete field from the SAME discrete 32-step sampling
trajectory, never independent random per-frame or per-joint noise levels.
The only time randomness is selecting a global index k from 0..31.
Training x_sigma=(1-sigma)x+sigma*epsilon, target v=epsilon-x.
The flow objective covers the cells actually updated on that discrete step.
Sampler integrates v*d_sigma, NOT v*d_p. Train and test use the same path.
Random anchor count, joint, frame and optional clean prefix are sampled at train.
Bidirectional axial spatial/temporal attention within the current window.

## Necessary controls

1. uniform noise schedule + identical hard anchors;
2. time-only increasing delay + identical hard anchors;
3. joint-time valley + identical hard anchors;
4. final-only clamping at inference (separately labelled distribution mismatch);
5. graph-distance / temporal-distance ablation if the first comparison helps.

Match architecture, parameters, training examples, optimizer, seed, training
steps, control masks, starting noise, inference network evaluations and hardware.
Zero anchor error alone cannot count as evidence for the valley hypothesis.
Report held-out free-joint error, bone length error, temporal derivatives and
generation quality in addition to anchor mean/max error and tolerance success.

## Data and official evaluation

Sources: https://github.com/AlayaLab/FloodDiffusion and
https://arxiv.org/html/2512.03520v1 . Archive SHA recorded in third_party.
Assets: ShandaAI/FloodDiffusionDownloads on Hugging Face. Use the publisher's
preprocessed features; retain its dataset notices. No public re-upload of data.
Use train/val/test splits and reject overlap, including HumanML mirrored ids.
Avoid the upstream default test_min as a final benchmark. Do not tune on test.

Official measures: HumanML3D evaluator FID, R-Precision@1/2/3 (groups of 32),
Matching score / MM-Dist (text-motion embedding distance), Diversity (300 pairs).
MM-Dist is not the multiple-samples-per-prompt multimodality metric.
Use frozen official evaluator weights only for evaluation, not training.
For XYZ output, deterministic XYZ->263 feature conversion must be validated by
ground-truth roundtrip before trusting embedding metrics. Report any conversion
floor. Original 263D features and reconstructed features are separate references.

BABEL: multi-action prompt changes; PJ and AUJ around transitions. The public
repository snapshot does not contain a PJ/AUJ implementation. Any own jerk
implementation must state axes, finite difference, units, transition window and
normalization; do not assert numerical equivalence to paper values until matched.
The paper's real-motion PJ=1.100 and AUJ=41.20 are reference numbers, not ours.
Latency: synchronized GPU timing, warm-up, first output latency, p50/p95 per
emission, sustained FPS, window size, lookahead and number of function evaluations.
Offline throughput is not streaming latency. Future anchors must be available
inside the declared lookahead; already emitted frames are immutable.

## Resource stages and spending gates

Reserve: $30 setup/data/pilots; $200 primary training; $150 seed/ablation repeats;
$50 BABEL/streaming evaluation; $70 storage, failures and contingency.
This is a cap, not a target spend. Stage 1 starts only after CPU tests pass.
Pilot: short real-data runs for the three schedules, validation-only comparisons.
Then profile a full training step and select a practical training budget.
Train three seeds for publishable comparisons only after pilot is stable.
Checkpoints, data and logs persist on a private Runpod network volume. Retrieve
results locally before terminating compute. Track resource ids/rates/timestamps.
Do not leave an unattended GPU running without a confirmed external cost guard.

## Approval gate (user instruction, 2026-09-23)

Before each paid GPU run, submit a detailed model/protocol/cost report and wait
for explicit approval. Budget authorization does not replace this approval.
No new paid GPU work is authorized until report v1 is approved. See AGENTS.md.

## Status

Core implementation has passed six CPU tests. Data/evaluation/training integration
has not run on real data. A 4090 pod was used ONLY for setup and partial downloads,
then terminated at the user's approval-gate instruction. No training was started.
No benchmark result exists. Neither improvement nor real-time generation has
been established. Private network volume remains; dataset downloads are partial.
