# Trustworthy Offline RL via Differential Privacy and Machine Unlearning

This project is a compact research implementation of privacy-preserving offline reinforcement learning for regulated, high-stakes decision systems. Healthcare trajectories, financial interaction logs, and other longitudinal records may encode sensitive membership and behavioural signals; a model trained on those logs needs both measurable privacy loss and a deletion path that can be audited when a data subject invokes erasure rights.

The framework combines trajectory-level Differential Privacy (DP), Conservative Q-Learning (CQL), influence-function unlearning, and SISA-style episode sharding. It is designed as a transparent implementation scaffold for discussing GDPR Article 17 deletion workflows, HIPAA-aware minimisation, and the practical tension between offline RL utility, privacy noise, and post-training removal requests.

## Pipeline

```mermaid
flowchart LR
    A[Offline Log Data<br/>Episode Trajectories] --> B[Trajectory DP-Training<br/>CQL + RDP Accounting]
    B --> C[Private Policy and Critics]
    C --> D[Unlearning Request<br/>Delete Episode tau_k]
    D --> E[LiSSA Influence Update<br/>or SISA Shard Retrain]
    E --> F[Certified Deletion Report<br/>MIA and Privacy Audit]
```

## What Is Implemented

- `optimizer.py` implements a bespoke PyTorch `TrajectoryDPSGD` optimiser. It sums transition losses for each episode, computes an explicit raw tensor gradient for that trajectory, clips the trajectory gradient to a phase-dependent norm `C`, injects Gaussian noise, and records an RDP-compatible budget estimate.
- `models.py` implements a compact continuous-action CQL agent. Its privacy-aware regulariser penalises positive out-of-distribution critic gaps more strongly when DP noise is present.
- `unlearning.py` implements LiSSA recursion for influence-function updates, a Gaussian noise interface for privatised unlearning corrections, and `EpisodeShardManager` for SISA-style episode-preserving deletion.
- `main.py` runs a synthetic offline RL workflow that trains with trajectory DP, deletes an episode, applies LiSSA unlearning, reports affected shards, and compares a threshold Membership Inference Attack (MIA) before and after deletion.

## Mathematical Foundations

Trajectory-level adjacency treats one full episode as the privacy unit. The trajectory sensitivity target is

$$
\Delta = \max_{\tau, \tau'} \left\| \nabla \ell(\tau) - \nabla \ell(\tau') \right\|_2.
$$

The optimiser therefore clips a gradient after transitions from the same trajectory have been aggregated. Standard transition-level DP-SGD instead clips each transition contribution independently, which protects a smaller adjacency unit and can understate the exposure of longitudinal logs.

For deletion auditing, the intended unlearning certificate is expressed as

$$
d_{\mathcal{TV}}\left(\mathcal{M}(\mathcal{D}), \mathcal{M}(\mathcal{D} \setminus \tau_k)\right) \leq \epsilon.
$$

LiSSA approximates the second-order removal direction

$$
H_\theta^{-1}\nabla_\theta \ell(\tau_k),
$$

without materialising a dense Hessian. SISA complements this local correction by keeping all transitions from an episode inside one shard, so a deletion request identifies bounded retraining work.

## Architecture

```text
trustworthy-offline-rl-via-dp/
  main.py        Synthetic training, deletion, RDP, and MIA CLI demo.
  models.py      ABC-backed offline RL API and privacy-aware CQL model.
  optimizer.py   Trajectory aggregation, adaptive clipping, Gaussian DP noise.
  unlearning.py  LiSSA influence updates, privatised updates, SISA sharding.
  utils.py       Episode tensors, synthetic data, and MIA simulation helpers.
```

## Quick Start

Create an environment with PyTorch available, then run from this directory:

```bash
python main.py --steps 8 --episodes 18 --delete-episode 3
```

The CLI prints the early and late clipping phases, an RDP budget estimate, the SISA shard IDs that need retraining for the deletion request, and the MIA report:

```text
step=00 clip=1.30 ...
step=07 clip=0.80 ...
rdp_budget={...}
deleted_episode=3 sisa_retrain_shards=[0]
mia_report={...}
```

To simulate a stronger deletion workload:

```bash
python main.py --steps 16 --episodes 36 --episode-batch-size 8 --shards 6 --delete-episode 11
```

### Smoke Test and Dynamic Visualisation

Use the compact smoke test to verify the core contract quickly:

```bash
python main.py --steps 8 --episodes 18 --delete-episode 3
```

For a 16-step multidimensional audit run, first run genuine training and write the latest metrics log:

```bash
python main.py --steps 16 --episodes 36 --episode-batch-size 8 --shards 6 --delete-episode 11
```

Then render the high-resolution figures from that exact experiment log:

```bash
python visualise.py --metrics-json results/plots/latest_visualise_metrics.json
```

`main.py` writes a reusable metrics log, and `visualise.py` only reads that log before rendering:

```text
results/plots/latest_visualise_metrics.json
results/plots/plot_privacy_utility.png
results/plots/plot_unlearning_margin.png
results/plots/plot_utility_tradeoff.png
```

To regenerate figures from a saved experiment log without rerunning training:

```bash
python visualise.py --metrics-json results/plots/latest_visualise_metrics.json
```

To use a named metrics log for a sweep, move or archive the JSON written by `main.py`, then pass it explicitly:

```bash
python visualise.py --metrics-json results/plots/metrics_steps16_episodes36.json
```

To run the full 16-step workflow and regenerate all evaluation figures in sequence:

```bash
python main.py --steps 16 --episodes 36 --episode-batch-size 8 --shards 6 --delete-episode 11 && python visualise.py --metrics-json results/plots/latest_visualise_metrics.json
```

### Real-World Simulation Ablations

The synthetic generator now uses sinusoidal state factors, cosine interactions, multidimensional drift, and a non-linear reward surface. This makes privacy noise and clipping choices more visible in the reported utility gap, MIA margins, and RDP curve.

Experiment A stresses the privacy-utility frontier with strong and weak noise:

```bash
python main.py --steps 32 --noise-multiplier 2.5 --state-dim 12 --write-metrics results/plots/high_priv.json && python visualise.py --metrics-json results/plots/high_priv.json --output-suffix high_priv
```

```bash
python main.py --steps 32 --noise-multiplier 0.05 --state-dim 12 --write-metrics results/plots/low_priv.json && python visualise.py --metrics-json results/plots/low_priv.json --output-suffix low_priv
```

Experiment B focuses on unlearning robustness for a specified deletion request:

```bash
python main.py --delete-episode 15 --write-metrics results/plots/unlearn_test.json && python visualise.py --metrics-json results/plots/unlearn_test.json --output-suffix unlearn_final
```

Experiment C compares adaptive and static clipping while preserving separate plot artefacts:

```bash
python main.py --steps 32 --clip-schedule adaptive --write-metrics results/plots/clip_adaptive.json && python visualise.py --metrics-json results/plots/clip_adaptive.json --output-suffix clip_adaptive
```

```bash
python main.py --steps 32 --clip-schedule static --static-clip-norm 1.0 --write-metrics results/plots/clip_static.json && python visualise.py --metrics-json results/plots/clip_static.json --output-suffix clip_static
```

### Domain-Agnostic IQL Runs

The CLI can now select both the data domain and offline RL algorithm. `PrivacyAwareIQL` avoids CQL-style OOD action sampling and instead uses in-sample expectile regression, which reduces gradient variance when trajectory-level DP noise is strong.

Medical experiments are specified around MIMIC-III, the Medical Information Mart for Intensive Care. In the intended protected-data setting, sepsis treatment trajectories are grouped by patient or ICU stay, clinical vitals and laboratory measurements become observations, and medication or intervention records become actions under the `EpisodeBatch` contract.

Financial experiments follow the FinRL portfolio optimisation interface. Asset-level price movement, volume, and indicator logs are treated as sequential trading histories, while execution strategy and alpha signals are protected as episode-level privacy units.

```bash
# Medical Run and Visualise
python main.py --domain medical --algo iql --noise-multiplier 1.5 --write-metrics results/plots/medical_iql.json
python visualise.py --metrics-json results/plots/medical_iql.json --output-suffix medical_iql
```

```bash
# Financial Run and Visualise
python main.py --domain financial --algo iql --noise-multiplier 1.5 --write-metrics results/plots/financial_iql.json
python visualise.py --metrics-json results/plots/financial_iql.json --output-suffix financial_iql
```

When no protected source file is supplied, `RealWorldTrajectoryLoader` produces schema-compatible domain proxies. With `--data-path`, it parses patient-level ICU trajectories or asset-level trading logs into the same `EpisodeBatch` contract used by the synthetic engine.

### Cross-Domain Comparative Analysis

The same privacy and unlearning machinery is deliberately reused across domains. This keeps the adjacency unit explicit while allowing domain-specific observations, actions, and rewards to vary.

| Domain | Adjacency Unit | Core Sensitive Data | Expected Utility $\Delta J$ |
| --- | --- | --- | --- |
| Medical | Patient trajectory | Clinical vitals and laboratory results | Lower variance via IQL in-sample bias |
| Financial | Trading asset history | Execution strategy and alpha signals | Stable returns under DP weight noise |

### Empirical Observations

The ablation suite exposes a clear privacy-utility frontier. Stronger Gaussian trajectory noise gives a markedly smaller RDP-derived privacy budget, but it also widens the expected-return degradation measured by the noisy policy proxy.

| Experiment | Noise multiplier | Final $\varepsilon$ | Utility gap $\Delta J$ | Interpretation |
| --- | ---: | ---: | ---: | --- |
| High Privacy | $\sigma=2.5$ | $\approx 3.22$ | $\approx 5.07$ | Stronger protection, higher utility loss |
| Low Privacy | $\sigma=0.05$ | $\approx 999.17$ | $\approx 2.99$ | Baseline utility, weak privacy protection |

These figures are consistent with the expected Bellman-side pressure introduced by privacy noise. In CQL, the critic is already deliberately pessimistic outside the logged support; when DP perturbations are enlarged, the empirical Bellman target remains contractive in form, yet its value-function optimisation landscape becomes noisier and less sharply aligned with the behaviour data. The observed increase in $\Delta J$ is therefore not an incidental plotting artefact: it is a measurable expression of the trade-off between trajectory indistinguishability and stable value estimation.

The clipping ablation further separates mechanism design from privacy accounting. Adaptive clipping follows a decaying schedule, beginning at $C=1.3$ and settling at $C=0.8$, whereas static clipping remains at $C=1.0$. The resulting curves make the optimisation schedule auditable without conflating it with the noise multiplier itself.

### Visualisation Gallery and Research Questions

The following artefacts are generated by `visualise.py` from JSON logs written by `main.py`. Each image answers a distinct empirical question while preserving the exact metrics that produced it.

| Figure | Research question |
| --- | --- |
| `results/plots/plot_privacy_utility_high_priv.png` | How quickly does cumulative $\varepsilon$ grow when strong privacy noise is imposed? |
| `results/plots/plot_unlearning_margin_high_priv.png` | Does strong privacy noise compress or shift MIA margins after unlearning? |
| `results/plots/plot_utility_tradeoff_high_priv.png` | How much utility is sacrificed when strong DP weight perturbation is applied? |
| `results/plots/plot_privacy_utility_low_priv.png` | How large does $\varepsilon$ become when the mechanism is nearly clean? |
| `results/plots/plot_unlearning_margin_low_priv.png` | What MIA margin structure remains when privacy noise is deliberately weak? |
| `results/plots/plot_utility_tradeoff_low_priv.png` | What utility level acts as the low-noise baseline for comparison? |
| `results/plots/plot_privacy_utility_unlearn_final.png` | What privacy budget is associated with the deletion-focused robustness run? |
| `results/plots/plot_unlearning_margin_unlearn_final.png` | Does the LiSSA correction move membership evidence towards post-deletion indistinguishability? |
| `results/plots/plot_utility_tradeoff_unlearn_final.png` | What utility cost accompanies the targeted deletion request? |
| `results/plots/plot_privacy_utility_clip_adaptive.png` | Does adaptive clipping produce the intended two-phase clipping trajectory? |
| `results/plots/plot_unlearning_margin_clip_adaptive.png` | How do MIA margins behave under the adaptive clipping schedule? |
| `results/plots/plot_utility_tradeoff_clip_adaptive.png` | What utility profile is obtained under adaptive trajectory clipping? |
| `results/plots/plot_privacy_utility_clip_static.png` | Does static clipping keep the clipping norm fixed across the full run? |
| `results/plots/plot_unlearning_margin_clip_static.png` | How do MIA margins compare when clipping is not phase-adaptive? |
| `results/plots/plot_utility_tradeoff_clip_static.png` | What utility profile is obtained under a fixed clipping norm? |

The dynamic plots summarise:

```text
step-wise RDP epsilon and clipping norm
before/after MIA margin distributions
logged-return vs DP-policy proxy utility trade-off
```

## Training and Deletion Flow

1. Offline episodes are generated or loaded as `EpisodeBatch` objects.
2. CQL computes transition objectives inside each episode.
3. `main.py` sums each episode objective before passing one scalar per episode to `TrajectoryDPSGD`.
4. The optimiser uses raw tensor gradients to clip each trajectory, inject Gaussian noise, and release an averaged update.
5. A deletion request selects one episode. LiSSA estimates its influence removal direction with an optional private noise mechanism.
6. `EpisodeShardManager` identifies the shard whose retraining and policy distillation workflow would be refreshed under SISA.
7. The MIA simulation compares membership leakage signals before and after unlearning.

## Compliance-Oriented Notes

- DP is a risk-control mechanism, not a blanket legal certification. The RDP estimate in this scaffold is intentionally visible so production accounting choices can be reviewed.
- GDPR Article 17 motivates a deletion workflow for user-contributed trajectories. Machine unlearning reduces the need to retrain every model from scratch, while shard boundaries make retraining scope auditable.
- HIPAA and financial compliance programmes still require data governance, access control, retention policies, provenance, and evaluation on the real deployment threat model.

## Extension Points

- Replace the synthetic episodes with healthcare or finance log loaders that preserve episode IDs.
- Swap the compact CQL critic for an IQL actor/value decomposition while retaining the trajectory optimiser contract.
- Add a production-grade subsampled Gaussian RDP accountant and deletion certificates backed by empirical retraining comparisons.
- Retrain only affected SISA shards and distil the shard ensemble into a serving policy via `EpisodeShardManager.distillation_loss`.
