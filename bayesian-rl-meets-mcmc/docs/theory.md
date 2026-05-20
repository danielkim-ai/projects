# Theoretical Anchors

## PAC-Bayes Bounds and Bayesian Regret

This note records the conceptual bridge between PAC-Bayesian generalisation theory and Bayesian regret for posterior-aware reinforcement learning. It is intended as an internal theoretical scaffold rather than a complete derivation.

## PAC-Bayesian View

PAC-Bayes theory studies the performance of a stochastic predictor drawn from a posterior distribution $Q$ over hypotheses, relative to a prior distribution $P$. In the present setting, the hypothesis corresponds to policy and value parameters $\theta$, and the posterior may be represented by an MCMC sampler, a variational family, or a hybrid approximation.

A canonical PAC-Bayes inequality relates the true risk of $Q$ to its empirical risk and its divergence from the prior:

$$
\mathrm{KL}
\left(
\widehat{L}(Q)
\,\|\, L(Q)
\right)
\leq
\frac{
\mathrm{KL}(Q \,\|\, P) + \log \frac{2\sqrt{n}}{\delta}
}{n}.
$$

Here, $\widehat{L}(Q)$ denotes empirical loss, $L(Q)$ denotes population loss, $n$ is the number of observed samples or trajectories, and $\delta$ is the confidence tolerance. The term $\mathrm{KL}(Q \,\|\, P)$ formalises the price of posterior adaptation: a posterior that moves too far from its prior must earn that movement through empirical evidence.

For reinforcement learning, the immediate complication is that trajectories are adaptively collected rather than independently sampled. Nonetheless, the PAC-Bayesian perspective remains valuable because it frames posterior concentration as a measurable trade-off between fit and complexity.

## Bayesian Regret View

Bayesian regret measures expected performance shortfall relative to the optimal policy under the true environment parameter, integrated over the prior. For horizon $T$, a common expression is

$$
\mathrm{BayesRegret}(T)
=
\mathbb{E}_{M \sim P}
\left[
\sum_{t=1}^{T}
\left(
V^{\star}_{M}(s_t)
-
V^{\pi_t}_{M}(s_t)
\right)
\right],
$$

where $M$ is the latent Markov decision process sampled from the prior, $V^{\star}_{M}$ is the optimal value function, and $V^{\pi_t}_{M}$ is the value of the policy deployed at time $t$.

Bayesian regret is therefore sensitive to how well the agent maintains and exploits epistemic uncertainty. A point-estimated actor-critic method may act as though posterior mass has collapsed prematurely, thereby inducing under-exploration and brittle value estimates.

## Relationship

The connection between PAC-Bayes bounds and Bayesian regret lies in posterior control. PAC-Bayes bounds penalise posteriors that depart excessively from priors without sufficient empirical support, while Bayesian regret penalises agents whose posterior uncertainty is not operationally useful for decision-making.

In this project, the posterior distribution $Q(\theta)$ is not merely a regulariser. It is an object of control:

$$
Q(\theta)
\approx
p(\theta \mid \mathcal{D})
\propto
p(\mathcal{D} \mid \theta)p(\theta).
$$

If $Q$ is calibrated, then policy updates can marginalise over plausible parameter settings rather than committing to a fragile point estimate. This suggests a route by which PAC-Bayesian complexity control may indirectly improve Bayesian regret: posterior distributions that are neither over-dispersed nor over-concentrated should support exploration policies that better reflect the remaining uncertainty.

## Implication for MCMC-VI Reinforcement Learning

The MCMC-VI framework can be interpreted as an attempt to preserve the statistical discipline of PAC-Bayes while improving the decision quality captured by Bayesian regret. Variational inference offers a tractable optimisation target, typically through

$$
\mathcal{L}_{\mathrm{ELBO}}(\phi)
=
\mathbb{E}_{q_{\phi}(\theta)}
\left[
\log p(\mathcal{D} \mid \theta)
\right]
-
\mathrm{KL}
\left(
q_{\phi}(\theta)
\,\|\, p(\theta)
\right),
$$

whereas SGLD or SGHMC can periodically correct or enrich the posterior approximation through stochastic sampling dynamics.

The working hypothesis is that posterior estimation quality mediates sample efficiency: better calibrated uncertainty should reduce the number of environment interactions required to identify useful policies, particularly when rewards are sparse, data are limited, or value targets are unstable.

## Epistemic and Aleatoric Uncertainty

The Bayesian machinery in this project targets epistemic uncertainty: uncertainty over policy, critic, and hyperparameter values that is reducible as more rollout data become available. It does not remove aleatoric uncertainty, which arises from irreducible environment stochasticity. This distinction is central. A calibrated posterior should contract under informative data, whereas irreducible transition or reward noise should remain represented in predictive variance.

## Hybrid Failure Modes

The hybrid MCMC-VI regime is not automatically superior. Recalibration may worsen performance if SGLD noise is too large relative to the policy-gradient signal, if the replay or rollout distribution is stale, or if the posterior target is itself non-stationary because the policy continually changes the data-generating process. Effective sample size should therefore be interpreted relative to the RL batch size, not as an isolated MCMC diagnostic.

The principal open questions are:

- whether posterior drift reflects genuine online Bayesian updating or concept drift induced by changing policies;
- whether the PAC-Bayes complexity term remains practically informative in continuous action spaces;
- whether SG-MCMC correction improves tail calibration without destroying the short-horizon stability obtained by variational updates.
