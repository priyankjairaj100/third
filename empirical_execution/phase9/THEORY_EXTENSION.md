# Realizable ridge heads force feature-scale repair memory

This is a new theorem addendum. It does not modify the frozen theory, numerical targets, software, or empirical results. The construction below is mathematical; its small verification fixtures are algebraic checks, not empirical datasets.

## Result and why it strengthens the existing stack

The previous query-rank theorem permits arbitrary independent record statistics. That theorem cannot justify a quadratic-in-feature-dimension memory lower bound for actual ridge moments. The existing binary-label construction avoids that problem, but establishes only one hidden bit per potentially admitted example. The later joint-span state uses `O(E(d+q))` numerical coordinates, where `E` is the eligible population, `d` the feature dimension, and `q` the response dimension. It did not have a realizable lower bound with that feature/response dependence.

The result here supplies that missing worst-case comparison. A fixed-margin cosine curator and the ridge learner use the **same unit features**. All forgotten blockers have fixed public payloads. Only previously excluded retained candidates vary. Moving the deletion budget from `b` to `b+1` changes the exact model-query summary requirement from zero private coordinates to exactly

\[
             m(d-1+q)
\]

real coordinates, among continuous encoders on the specified open parameter domain. An independent packing argument gives a finite-bit lower bound of order `m(d+q)` at a sufficiently small fixed parameter error and fixed per-query failure probability. Neither result is an arbitrary-real encoding claim, a total-byte optimality theorem, or a statement about every natural corpus.

The exact upper bound is a **model-query summary**: it answers all prescribed counterfactual requests from the initial state. We do not call it a canonical deletion-clean state implementation. The established joint-span/canonical implementations supply a different, stronger state-repair contract and retain their existing storage, arithmetic, precision, and metadata qualifications.

## 1. Contract

There are `N=1+b+2m` records, with `b>=1`, `m>=2`, ambient feature dimension `d>=3`, and `q>=1` response coordinates. The fixed ordered curator keeps a retained record exactly when all its earlier **raw** neighbors have been deleted. Edges use the strict unit-cosine threshold `tau=2/5`.

The prescribed learner on a selected set `S` of size `n>0` minimizes

\[
 L_S(W)=\frac1{2n}\sum_{v\in S}\|z_v^T W-y_v^T\|_2^2
                   +\frac\lambda2\|W\|_F^2,\qquad \lambda>0.
\]

It returns the exact zero `d` by `q` head for the empty set. Queries are all identifier sets of cumulative size at most the stated budget, interpreted counterfactually from the initial corpus. The lower bound already follows from a smaller fixed set of single counterfactual probes. It therefore also applies to a sequential service whose initial state must support every allowed first request. Reusing/cloning that initial state in a mathematical distinguishability proof is not a requirement that the service permit operational rollback.

Public information consists of identifiers, order, the fixed graph, the construction parameters, the candidate cap centers and coordinate frames, and the fixed zero-label blocker records. It does **not** include the actual varying candidate features or responses. The state includes every candidate-dependent cache, model, feature table, remote store, ticket, and retained private random seed. No retained-candidate reread or free dataset-dependent side channel is allowed. A query may supply the full forgotten payloads; the probes used below delete only fixed public records, so this access cannot weaken the lower bound.

The continuous-coordinate theorem concerns a deterministic encoder into a fixed Euclidean space `R^s`, continuous as a function of the open real parameter chart. The decoder need not be continuous. Arbitrary discontinuous encoders are outside that theorem. The finite-bit theorem instead permits arbitrary encoders and randomization, but charges the complete private state to a message alphabet of size at most `2^B`.

## 2. A strict-margin family with genuinely private features

Choose public unit signatures `u_1,...,u_m` in a `(d-2)`-dimensional subspace with

\[
                  |u_i^T u_j|\le1/6\quad(i\ne j).
\]

Let `e_0,e_1` be orthogonal to that subspace. In priority order, use anchor `a`, common blockers `s_1,...,s_b`, private blockers `p_1,...,p_m`, and candidates `w_1,...,w_m`. Define public fixed records and candidate centers by

\[
\begin{split}
 a&=e_0,&s_j&=\tfrac8{17}e_0+\tfrac{15}{17}e_1,\\
 p_i&=\tfrac8{17}e_0+\tfrac{15}{17}u_i,&
 w_i^0&=\tfrac8{17}e_1+\tfrac{15}{17}u_i.
\end{split}
\]

All are unit vectors. The rational coefficients are a convenient new realization of the existing activation-frontier graph, using the Pythagorean triple `(8,15,17)`. The relevant unperturbed inner products are:

| Pair | Value or upper bound | Required relation to `2/5` |
| --- | --- | --- |
| Anchor/common or private blocker | `8/17` | Above |
| Anchor/candidate center | `0` | Below |
| Common blocker/candidate center | `120/289` | Above |
| Matching private blocker/candidate center | `225/289` | Above |
| Nonmatching private blocker/candidate center | `75/578` | Below |
| Distinct private blockers or candidate centers | `203/578` | Below |
| Common/private blocker | `64/289` | Below |

The common blockers form a harmless clique. Set `r=1/100`. Candidate `i` may vary independently anywhere on the unit-sphere cap

\[
            w_i\in\mathbb S^{d-1},\qquad \|w_i-w_i^0\|_2<r.
\]

A fixed-to-varying inner product changes by less than `r`; one involving two varying candidates changes by less than `2r`. The tight common/candidate margin is `22/1445>r`, and the candidate/candidate nonedge margin is `141/2890>2r`. All other required margins are larger. Thus every feature choice in these independent caps has the same graph and

\[
                    B(w_i)=S_0\cup\{p_i\},\qquad
                    S_0=\{s_1,\ldots,s_b\}.
\]

In particular, the blocker feature `p_i` does not move when `w_i` moves. Varying both with the same hidden direction would leak that direction in the forgotten payload and invalidate the intended lower bound.

There are suitable signatures whenever `d-2` is sufficiently large compared with `log m`: independent normalized sign vectors have pairwise failure probability at most `2 exp(-(d-2)/72)`, so a union bound gives existence, for example, when `d-2>=ceil(72 log(2m(m-1)))`. Padding by zero coordinates allows larger `d`. Orthogonal signatures give the simpler alternative `d>=m+2`. The theorem is conditional on the signatures; it does not assert arbitrarily many candidates at fixed `d`.

## 3. Exact ridge heads identify full feature directions

Write

\[
 C=aa^T+2\lambda I_d,\qquad c=1+2\lambda,\qquad
 \beta_0=\frac\lambda{2c}.
\]

For each candidate choose `beta_i` independently in `(beta_0,2 beta_0)^q`, and set its actual response to

\[
 y_i=\bigl(1+w_i^T C^{-1}w_i\bigr)\beta_i.
\]

Every non-candidate response is zero. Since `1+w_i^T C^-1 w_i <= c/(2 lambda)`, every actual response coordinate lies strictly between zero and `1/2`. This is a coordinatewise bound; its Euclidean norm can grow as `sqrt(q)`. The feature and response parameters need not be probabilistically independent; the construction deliberately uses a smooth coupled response parameterization.

Initially only the zero-label anchor is selected, so the original head is exactly zero and carries no private information. At budget `b`, no candidate can be selected under any allowed deletion. Every target head is therefore zero, even if deleting the anchor admits zero-label blockers.

At budget `b+1`, let

\[
                         F_i=S_0\cup\{p_i\}.
\]

The selected set is exactly `{a,w_i}`. Conversely, if any candidate is selected by a request of size at most `b+1`, that request must equal its `F_i`; it cannot contain another identifier. The retained ridge normal equations are

\[
 (C+w_iw_i^T)W_i=w_i y_i^T.
\]

Sherman–Morrison gives the particularly useful identity

\[
                    W_i=C^{-1}w_i\beta_i^T.                 \tag{1}
\]

Equation (1) is an actual realizable ridge optimizer, not an arbitrary matrix response. From it,

\[
 w_i=\frac{CW_i[:,1]}{\|CW_i[:,1]\|_2},\qquad
 \beta_{ij}=\|CW_i[:,j]\|_2.
\]

Positivity removes the sign ambiguity. Thus the head identifies all `d-1` local unit-direction parameters and all `q` response parameters of that candidate.

## 4. Exact continuous-coordinate theorem

**Theorem 1.** Fix `b>=1`, `m>=2`, `q>=1`, `lambda>0`, and a dimension `d` admitting the signatures specified in Section 2; its displayed logarithmic dimension bound is sufficient. Fix the resulting public construction. Let the private features range over the cap chart described below and the private `beta` coordinates over their open intervals. At budget `b`, zero private coordinates suffice for all exact ridge-head queries. At budget `b+1`, the minimum fixed number of real coordinates in a continuous deterministic exact model-query encoder is exactly

\[
                             M=m(d-1+q).
\]

**Proof.** For each center choose a public orthonormal tangent frame `V_i` with `d-1` columns. Parameterize an open ball by

\[
 w_i(t_i)=\sqrt{1-\|t_i\|_2^2}\,w_i^0+V_i t_i,
                      \qquad \|t_i\|_2<r/2.
\]

The feature displacement is less than `r`, and `t_i=V_i^T w_i`. The product of these balls and the open `beta` intervals is an open subset of `R^M`. If two different parameter tuples had the same stored state, every `F_i` would produce the same exact head under the same fixed forgotten payloads. The recovery formulas above would force all their parameters to agree. The encoder must therefore be injective on this open domain.

There is no continuous injective map from an open subset of `R^M` to `R^s` for `s<M`: compose such a map with the coordinate inclusion into `R^M`; invariance of domain would make its image open, whereas that image lies in a proper coordinate subspace with empty interior. Hence `s>=M`. This is the standard topological dimension argument.

For the upper bound, store each `(t_i,beta_i)`. On `F_i`, recover `w_i` and use (1); on every other allowed request return zero. The encoder is continuous and uses exactly `M` coordinates. At budget `b`, always return zero. This proves the stated query-summary minimum. QED.

Any deterministic sequential exact service with a continuous initial encoder inherits the lower bound: it must answer each `F_i` as a possible first request. This does not turn the chart upper bound into a sequential canonical-state theorem. It also does not lower-bound an arbitrary discontinuous real encoder; such an encoder can hide unbounded information in one real number. Merely assuming a continuous decoder would not fix that issue.

For fixed `b`, `N=Theta(m)` and the high-budget eligible population is `E=N`. Thus `M=Theta(E(d+q))`. This matches the feature/response order of the existing `O(E(d+q))` compact state in a worst-case exact-coordinate sense. The constants, update work, decoder workspace, identifier/graph storage, and rational bit growth are outside that comparison. In particular, this result does not establish a physical-byte advantage over the compact eligible-payload baseline.

## 5. A separate finite-bit theorem at nonvanishing error

The continuous-domain obstruction alone says nothing about finite precision or approximate answers. The following independent packing argument provides a bit claim with explicit error scale.

For two parameter pairs, unit norms give the exact identity

\[
\|w\beta^T-w'\beta'^T\|_F^2
 =\|\beta-\beta'\|_2^2+(\beta^T\beta')\|w-w'\|_2^2.
\]

Since every `beta` coordinate is at least `beta_0`, and the largest eigenvalue of `C` is `c`,

\[
\|W-W'\|_F\ge\frac1c
\sqrt{\|\beta-\beta'\|_2^2+\beta_0^2\|w-w'\|_2^2}.       \tag{2}
\]

The chart above satisfies `||w(t)-w(t')|| >= ||t-t'||`, by orthogonally projecting the difference onto the tangent frame. Use the closed ball of radius `r/2` for this packing argument; it still lies strictly inside the allowed feature cap.

For `0<eta<r beta_0/(16c)`, a maximal `4c eta/beta_0`-separated packing of that ball has cardinality at least

\[
 L_w\ge\left(\frac{r\beta_0}{8c\eta}\right)^{d-1}.
\]

Indeed, maximality makes balls of the separation radius cover the original ball, and comparison of Euclidean volumes proves the count. A coordinate grid in `[beta_0,2 beta_0]^q` of spacing `4c eta` has

\[
                 L_\beta=\left(\left\lfloor
                        \frac{\beta_0}{4c\eta}\right\rfloor+1\right)^q
\]

points. These closed intervals are used only for the finite packing and obey the same response bounds. The Cartesian product has `L=L_w L_beta` heads separated by at least `4 eta`, by (2). Thus an answer within `eta` determines the candidate's packing symbol by nearest-head decoding.

**Theorem 2.** Let the state contain at most `B` private bits, allowing public randomness independent of all candidate parameters and fresh randomized repair. Suppose every fixed input in the finite product construction and every probe `F_i` is answered with Frobenius error at most `eta` with probability at least `1-delta`, where `0<=delta<1/2`. Then

\[
 B\ge m\bigl[(1-\delta)\log_2 L-h_2(\delta)\bigr].       \tag{3}
\]

It suffices to substitute the displayed lower bounds for `L_w` and `L_beta` into (3). In particular, for the stated error range,

\[
 B\ge m\bigl[(1-\delta)(d-1+q)-h_2(\delta)\bigr],
\]

and the stronger explicit logarithmic expression is available at smaller error. The tolerable `eta` depends on `lambda` and the fixed cap radius, but not on `m,d,q`. It can be numerically small; no claim of a practically large tolerance is made.

**Proof.** Give each candidate an independent uniform packing symbol `Z_i` in an alphabet of size `L`. Denote the state by `M_state` and public coins by `R`. On a successful answer, nearest-head decoding recovers `Z_i`. An optimal decoder based on `(M_state,R,i)` has at most the stated average error, even if repair uses fresh randomness. The elementary M-ary Fano bound yields

\[
 H(Z_i\mid M_{state},R)\le h_2(\delta)+\delta\log_2(L-1)
                         \le h_2(\delta)+\delta\log_2 L.
\]

Conditional subadditivity and independence give

\[
 B\ge I(Z_1,\ldots,Z_m;M_{state}\mid R)
   \ge m\log_2L-\sum_i H(Z_i\mid M_{state},R),
\]

which proves (3). There is no simultaneous-success assumption and no union bound over query outputs. The proof also works when correctness is required only on average over uniform symbols and a uniform probe index: apply Fano with the individual decoding errors and use concavity of binary entropy to average them. All forgotten probe payloads are fixed independently of these symbols. QED.

This is a finite-alphabet information lower bound over an explicitly separated real-valued construction. It does not assert that those packings are representable by the repository's fixed FP32 embeddings or that a matching finite-bit canonical repair implementation is available. Quantization must preserve the required margins and output separation before applying a fixed-format version. Fixed-accuracy bit lower bounds and exact real-coordinate upper bounds are different statements.

## 6. The feature obstruction already holds with public binary labels

The main parameterization isolates both feature and response information. Its feature-dimension conclusion does not require varying or feature-dependent labels.

**Corollary 3.** In the same graph/cap family, take one response coordinate, give every candidate the same public label `y_i=1`, and every blocker/anchor label zero. The exact continuous model-query state minimum is zero at horizon `b` and `m(d-1)` coordinates at horizon `b+1`. For the same randomized finite-bit contract, at

\[
                  0<\eta<\frac{r\lambda}{8c^2},
\]

there is a finite candidate packing with

\[
 L_w\ge\left(\frac{r\lambda}{4c^2\eta}\right)^{d-1},\qquad
 B\ge m\bigl[(1-\delta)\log_2L_w-h_2(\delta)\bigr].
\]

**Proof.** The hard head is `W=C^-1 w beta(w)`, with `beta(w)=1/(1+w^T C^-1 w)>=2 lambda/c>0`. Normalize `CW` to recover `w`. The injective continuous-encoder argument and chart storage give the exact dimension. For approximation, the same rank-one identity implies

\[
 \|W-W'\|_2\ge\frac{2\lambda}{c^2}\|w-w'\|_2.
\]

A chart packing of separation `2c^2 eta/lambda` has the displayed cardinality and head separation at least `4 eta`; Fano applies unchanged. QED.

All responses in this corollary are binary and public. Only candidate feature directions are private. Its `[0,1]` response class is separate from the main theorem's `<=1/2` construction. This does not assert that natural text embeddings actually fill the cap family.

## 7. What may be claimed

- Deletion-induced admissions can force memory proportional to feature plus response dimension **even for actual bounded-response ridge heads**, with the same strict cosine features used for curation and learning.
- The adjacent-budget phenomenon survives full forgotten-payload access because those payloads are deliberately fixed.
- The `O(E(d+q))` exact numeric scaling has a matching worst-case continuous-coordinate obstruction; it is not merely an improvement over an unnecessarily dense Gram implementation.
- The information obstruction persists at a fixed, explicitly bounded approximation scale, under randomized per-query correctness.

Do not claim arbitrary-real or byte optimality, a universal feature-dimension lower bound for a fixed corpus, constant-dimensional geometry for arbitrarily large `m`, a practical optimal lossy codec, privacy erasure, an adaptive-query upper bound, fixed-FP32 realization without a quantization argument, or semantic/NLP empirical evidence from this family.

The original query-rank theorem remains useful but has a different access model. In particular, with public fixed features, variable labels, and identifier-only requests, stacked ridge heads have a linear observable-label rank. If deleted labels arrive as free query side information, that rank by itself need not lower-bound storage: a single stored total plus the forgotten label answers scalar singleton-deletion sums. The present construction avoids that issue directly instead of broadening the old rank theorem.

## 8. Attribution and verification

The curation architecture extends the repository's integrated activation-frontier theorem and `round3_lower.md`; it does not rebrand that architecture as a new discovery. The new strengthening is the private-feature/response parameterization, fixed forgotten payloads, full-sphere ridge-head identifiability, and the resulting feature-scale state bound.

The mathematical tools are classical: Sherman–Morrison, invariance of domain, volumetric packing, and Fano/random-access information bounds. For the continuous-injection theorem see Allen Hatcher, *Algebraic Topology*, Theorem 2B.3, printed page 172: <https://pi.math.cornell.edu/~hatcher/AT/AT.pdf>. For a primary-authored account of Fano with side information and packing, see Jonathan Scarlett and Volkan Cevher, *An Introductory Guide to Fano's Inequality with Applications in Statistical Estimation*, arXiv:1901.00555v3, Theorem 1 and Sections 2.3 and 5.2: <https://arxiv.org/pdf/1901.00555>. The existing broad bounded-deletion storage literature, including Cherapanamjeri et al., COLT 2025, remains relevant: <https://proceedings.mlr.press/v291/cherapanamjeri25b.html>. This addendum does not claim to invent its proof techniques or certify exhaustive novelty.

Reproduce the retained exact-rational algebra checker with:

```bash
python3 empirical_execution/phase9/check_theory_extension.py --output empirical_execution/phase9/results/theory_extension_checks.json
```

It checks the strict graph geometry, all bounded deletion subsets in tiny cases, the fixed forgotten payloads, exact fresh ridge solves, the response bounds, full-sphere feature variation, the rank-one separation identity, local parameter rank, and packing constants. It binds its own source, this note, the LaTeX fragment, and the frozen sources it builds upon. It does not prove the continuous-domain topological or information-theoretic theorems by testing. The complete proofs are above and in `theory_extension.tex`.
