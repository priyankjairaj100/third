# Finite-word memory forced by counterfactual admissions

This addendum freezes no new empirical claim and changes no prior phase. It replaces the continuous-input assumption in Phase 9's worst-case witness with a finite alphabet of exactly stored FP32 feature vectors. The result concerns information needed to reproduce prescribed ridge heads. The construction is mathematical; bounded checks of it are proof/software fixtures, not synthetic empirical evidence.

## 1. Main result and scope

There is an explicit family with public binary labels, a fixed strict-cosine curator, and private sign bits contained in the same raw feature vectors used by the ridge learner. Increasing the record-deletion horizon from `b` to `b+1` changes the exact private state requirement from zero to **`mD` bits**. Here `m` is the number of potentially admitted candidates and `D` the dimension of their private feature block.

This is a finite-alphabet lower bound. It needs no continuous encoder, real-coordinate model, arbitrary independent Gram entries, or irrational input features. It allows all forgotten-record payloads to be supplied: the probes establishing the bound delete fixed public blockers.

Three distinct results follow:

1. The exact initial state needs and can attain `mD` private bits; after cumulative deletion `F`, the exact minimum is `e(F)D` conditional on the public deletion ledger, where `e(F)` counts candidates still reachable within the original horizon.
2. Expected squared **parameter** error has a sharp information–distortion lower bound and a finite-length query-summary coding upper bound using classical Bernoulli rate-distortion tools.
3. A sufficiently accurate numerical release still identifies every private bit. This is a conditional certificate statement, not a claim that a particular floating solver meets that certificate.

The fixed-format/scorer theorem has an explicit bounded domain: `k,D` are powers of four and `d=2+k+D<=2^20`. Scaling statements below are uniform inequalities on that admitted domain, not an unbounded-dimension asymptotic under one frozen floating-point format. The feature vectors are mathematical FP32 inputs; they are not claimed to be outputs of E5 or any other actual text encoder.

## 2. Public contract and private information

Fix integers `b>=1`, `m>=2`, powers of four `k,D>=1` with `d=2+k+D<=2^20`, and a fixed positive rational regularizer `lambda`. The public signature matrix has rows

\[
 u_i\in\{\pm1/\sqrt{k}\}^{k},\qquad
 |u_i^T u_j|\le1/6\quad(i\ne j).
\]

The theorem is conditional on this verified code property. Orthogonal Hadamard rows provide one concrete choice when `m<=k`. More generally, normalized independent sign vectors give such a code with positive probability whenever `k>=ceil(72 log(2m(m-1)))`; choose a power of four satisfying this and the total-dimension cap. This does not promise arbitrary `m` at fixed `k`.

There are `N=1+b+2m` records. Fix `N` distinct public identifier strings and a public priority seed independently of all private signs. Sort them using the frozen `phase3.reference_graph.stable_priority`, and assign the resulting positions to anchor `a`, common blockers `S={s_1,...,s_b}`, private blockers `p_1,...,p_m`, then candidates `w_1,...,w_m`. The stable ID tie break makes this construction independent of any assumption about hash collisions. This is an assignment of mathematical roles to a fixed public order, not a search over natural records or a provenance claim.

The only varying data are `mD` independent sign choices. Write

\[
 z_i\in\{\pm1/\sqrt D\}^{D},\qquad i=1,\ldots,m.
\]

Identifiers, role map, signatures, dimensions, regularization, labels, threshold, graph, and all blocker payloads are public. Actual candidate private signs/features are not public. Count every sign-dependent deployment head, cache, table, remote store, ticket, retained random seed, and usable earlier output in the state. No retained-candidate reread or uncharged sign-dependent side channel is available. Public randomness must be independent of the signs. An ordinary request may include its full forgotten payloads; the lower-bound probe payloads contain no private signs.

Throughout, a **`B`-bit state/message means an alphabet with at most `2^B` distinguishable values conditional on the public contract and public coins**. Fixed-capacity `B`-bit storage satisfies this convention. Any usable length, framing, or allocation choice must be charged; it cannot be an additional free data-dependent channel. Unrestricted variable-length strings of length at most `B` have `2^(B+1)-1` possible values and do not automatically satisfy this convention. The entropy arguments below specifically use `H(M|R)<=B`.

The learner is average-loss scalar ridge on the **stored raw vectors**, with no learner normalization:

\[
 L_T(\theta)=\frac1{2|T|}\sum_{v\in T}(x_v^T\theta-y_v)^2
                       +\frac\lambda2\|\theta\|_2^2.
\]

The empty selected set returns zero. The target is the exact rational optimizer of this objective on the stored FP32 values and public rational `lambda`. This target differs from prescribing the bit pattern produced by a particular FP32 or FP64 solver. Cosine normalization occurs only inside graph scoring; it does not overwrite the learner's stored vectors.

All results concern record deletion. Giving every mathematical record a distinct deletion owner makes singleton-source deletion formally equivalent, but this is not a claim about arbitrary source groups, genuine source metadata, or the empirical source-withdrawal arm.

## 3. Exactly stored features and the frozen scorer

Use orthogonal coordinate blocks `e_0`, `e_1`, `U` of length `k`, and `H` of length `D`. Put `alpha=1/128` and define

\[
\begin{split}
 a&=e_0,\\
 s_j&=\tfrac12e_0+\tfrac78e_1,\\
 p_i&=\tfrac12e_0+\tfrac78u_i,\\
 w_i&=\tfrac12e_1+\tfrac78u_i+\alpha z_i.
\end{split}                                                     \tag{1}
\]

Every candidate has public label one; every other record has public label zero. The unnormalized row norms are

\[
 \|a\|^2=1,\quad \|s_j\|^2=\|p_i\|^2=65/64,\quad
 \|w_i\|^2=65/64+1/16384=16641/16384=(129/128)^2.                 \tag{2}
\]

Because `k,D` are powers of four, their square roots are powers of two. All nonzero coordinates in (1) are dyadic, exactly representable **normal** FP32 values throughout the declared dimension range. Zeros are also exactly stored. Changing a private sign cannot change a row norm.

The curator uses the frozen FP64 ordered normalization/dot-product scorer, with the stored threshold

\[
 \tau=\operatorname{float64}(0.4)
      =3602879701896397/2^{53}
      =2/5+1/(5\,2^{53}).                                      \tag{3}
\]

The ideal normalized cosines obey these uniform bounds, for every private sign payload:

| Pair | Exact value or upper bound |
| --- | --- |
| Anchor/common or private blocker | `4/sqrt(65)` |
| Anchor/candidate | `0` |
| Common blocker/candidate | `448/(129 sqrt(65)) > .43` |
| Matching private blocker/candidate | `784/(129 sqrt(65))` |
| Nonmatching private blocker/candidate | At most `784/(774 sqrt(65))` |
| Distinct private blockers | At most `29/78` |
| Distinct candidates | At most `18563/49923 < .372` |
| Common/private blocker | `16/65` |

Common blockers have identical vectors, hence are mutually adjacent. All required edges exceed `.43`; all required nonedges are below `.372`. For candidate pairs the private contribution is bounded by `alpha^2 |z_i^T z_j|<=alpha^2`, so this statement is uniform over all `2^(mD)` private assignments.

Under IEEE binary64 round-to-nearest operations with the ordered evaluation in the frozen source, the companion scorer proof bounds the absolute error of every normalized score by

\[
 E_d=2\rho+\rho^2+\gamma(1+\rho)^2<2^{-32},\qquad
 \rho=\frac{2u}{1-u},\quad
 \gamma=\frac{(d+1)u}{1-(d+1)u},\quad u=2^{-53}.                 \tag{4}
\]

The stored squares and partial norm sums are exact in this family; all partial sums are dyadic multiples of `2^-34` with magnitude below two. The remaining square-root/division and ordered dot-product roundings are covered by (4). The bound assumes the stated operations, finite normal intermediates, and no unadvertised reassociation or altered arithmetic. It is a proof for the declared scorer, not just a finite execution comparison. There is no clipping operation in the frozen scorer.

The error in (4) is far smaller than both margins around (3). Consequently every private assignment yields the same frozen graph, including

\[
                       B(w_i)=S\cup\{p_i\}.                  \tag{5}
\]

Normalization for curation leaves the raw learner vectors in (1) unchanged. A learner that instead consumes the normalized graph vectors is a different problem and does not satisfy the formulas below.

## 4. Exactly observable finite private information

Define

\[
                t=16641/16384+2\lambda,\qquad n=mD.
\]

Initially only the zero-label anchor is selected. At any record-deletion budget `b`, no candidate can lose all `b+1` blockers. All selected labels remain zero, including when deleting the anchor admits zero-label blockers. Every exact head is zero.

At budget `b+1`, the request

\[
                         F_i=S\cup\{p_i\}
\]

selects exactly `{a,w_i}`. Conversely, any request of size at most `b+1` selecting a candidate must equal its `F_i`. Because `a` is orthogonal to every `w_i`, the exact ridge head is

\[
 \theta_i=\frac{w_i}{t}
 =\underbrace{\frac{\tfrac12e_1+\tfrac78u_i}{t}}_{\theta_i^{\rm pub}}
                  +\frac{\alpha}{t}z_i.                       \tag{6}
\]

Its `H` coordinates are `±alpha/(t sqrt(D))`; their signs reveal the candidate's entire private `D`-bit word.

**Theorem 1 — exact finite-alphabet minimum.** On this fixed public family, a deterministic exact model-query summary requires zero private bits at horizon `b` and exactly `mD` private bits at horizon `b+1`. The lower bound also holds for a sequential service's initial state and for randomized services that are exactly correct almost surely for every fixed input and request.

**Proof.** There are `2^(mD)` assignments. If two assignments shared an exact state, they differ in at least one candidate word, whose fixed public request `F_i` has two different target heads by (6). The state could not answer both. Therefore at least `2^(mD)` states, or `mD` bits, are necessary. Store all signs to attain the bound: on `F_i` reconstruct (6), and on every other valid query return zero. Low horizon needs no signs. The randomized zero-error claim also follows by setting the distortion to zero in Theorem 3 below; equivalently, finite input/query alphabets permit fixing public randomness outside the common null failure set. A sequential service must support each `F_i` as a possible first request. QED.

This counts private information, not total serialized bytes. The fixed graph, signatures, ID ledger, code, arithmetic workspace, and alignment/padding have real storage costs. In particular, an `mD`-bit information minimum is not a claim that a complete service occupies only `mD` bits.

Choosing `D>=k`, for example `D=k`, makes `D>=(d-2)/2`. Thus for fixed `b` the bound is proportional to `N*d` over that admitted subfamily. When most dimensions instead belong to the public signature block, the precise lower bound remains `mD`; it must not be inflated to `md`. On fixed-width 32-bit private storage words the elementary capacity consequence is at least `ceil(mD/32)` words, with all remaining metadata and workspace additional.

## 5. The exact minimum after prior deletions

Keep the original horizon `K=b+1`; it is never reset after a request. For an admissible cumulative deleted set `F`, define

\[
 I(F)=\{i:w_i\notin F,\ |B(w_i)\setminus F|\le K-|F|\}
     =\{i:F\subseteq S\cup\{p_i\}\},\qquad e(F)=|I(F)|.       \tag{7}
\]

The second equality uses `|B(w_i)|=K`; it automatically implies that the candidate survives. This includes currently selected candidates when no remaining deletion is possible: reproducing the current head is still a valid empty continuation.

**Theorem 2 — conditional retained-state minimum.** Fix any admissible cumulative deleted set `F` and demand exact answers for every private assignment and every allowed continuation. Conditional on this fixed public construction and ledger, the exact minimum is `e(F)D` private bits. All retained sign-dependent auxiliary information, including any usable earlier output, counts toward that state. This is a worst-case fixed-`F` statement; it does not assert that conditioning on the realized requests of a particular adaptive policy leaves the private signs' entropy unchanged.

**Proof.** If `i` is in `I(F)`, complete the request with `B(w_i)\F`. It fits the remaining horizon and yields the original hard probe `F_i`, recovering its private word. Vary the eligible words independently and fix all other private words; the state must distinguish `2^(e(F)D)` possibilities. Whenever `e(F)>0`, `F` contains only fixed public blockers, so its forgotten payloads supply none of these bits. For sufficiency retain exactly the words indexed by `I(F)` and use (6) for any current/future hard query. All other heads are zero. Since `F` only grows under the fixed horizon, `I(F')` is a subset of `I(F)` for every continuation; discarded words can never become relevant again. QED.

For this family, deleting only common blockers leaves all `m` words necessary. Deleting one private blocker can reduce the reachable population to its single candidate. Deleting the anchor, a candidate, or two distinct private blockers leaves no candidate reachable within the original budget. These are information statements conditional on `F`; the actual deletion ledger and its validation are additional public-state costs. The separately implemented Phase 10 canonical codec supplies the concrete bit packing, atomic updates, membership/horizon checks, padding convention, and retained-state comparison. The lossy coding result below does not substitute for that exact sequential contract.

## 6. Expected squared parameter error: a sharp information curve

For `0<=v<=1/4`, define

\[
 q(v)=\frac{1-\sqrt{1-4v}}2,\qquad
 R_{\rm sq}(v)=1-h_2(q(v));
\]

put `R_sq(v)=0` for `v>=1/4`. Logs in information formulas are base two.

**Theorem 3 — expected parameter-fidelity bound.** Let a message contain at most `B` private bits. Public random coins independent of the signs and fresh repair randomization are permitted. If for every fixed private assignment and every fixed **hard probe** `F_i`,

\[
                 \mathbb E\|\widehat\theta_i-\theta_i\|_2^2
                       \le\varepsilon,
\]

then

\[
 B\ge nR_{\rm sq}(v),\qquad
 v=\frac{t^2\varepsilon}{4\alpha^2},\qquad n=mD.              \tag{8}
\]

For the lower bound it suffices to average the loss over uniformly independent private signs and a uniform index among the `m` hard probes. Averaging over all valid deletion sets is insufficient because the many zero-target requests could dilute the error.

**Proof.** Write private signs as `2Y_ij-1`, where `Y_ij` are independent uniform bits. Project an arbitrary answer onto `H` and define

\[
 \widehat Y_{ij}=\frac12\left(1+
          \frac{t\sqrt D}{\alpha}\widehat\theta_i[H_j]\right).
\]

Clipping these estimates to `[0,1]` cannot increase their squared error. Before clipping,

\[
 \frac1D\sum_j(\widehat Y_{ij}-Y_{ij})^2
  =\frac{t^2}{4\alpha^2}
       \|P_H(\widehat\theta_i-\theta_i)\|_2^2
  \le\frac{t^2}{4\alpha^2}\|\widehat\theta_i-\theta_i\|_2^2.
\]

Let `M` be the message and `R` the public coins. The posterior mean `p_l=Pr(Y_l=1|M,R)` minimizes conditional bitwise squared error, even compared with randomized decoding. Thus the average posterior variance is at most `v`.

The function `g(s)=h_2((1-sqrt(1-4s))/2)` is increasing and concave on `[0,1/4]`. To check the latter, set `s=q(1-q)` with `q<=1/2`; the relevant derivative sign is that of `-u+1/u+2 ln u<=0`, where `u=(1-q)/q>=1`. This follows because `u-1/u-2 ln u` vanishes at one and has derivative `(u-1)^2/u^2>=0`. Since `h_2(p)=g(p(1-p))`, conditional subadditivity and Jensen give

\[
 H(Y\mid M,R)\le\sum_l\mathbb E h_2(p_l)
                  \le n g(v),\qquad
 B\ge I(Y;M\mid R)\ge n[1-g(v)].
\]

For `v>=1/4` use the trivial bound zero. QED.

The zero-information threshold is exactly

\[
                         \varepsilon_0=\alpha^2/t^2.          \tag{9}
\]

Returning only the known public component `theta_i^pub` on every hard query attains this squared error for every sign assignment, with no private state. Any fixed `varepsilon<varepsilon_0` forces a positive dimension-proportional information rate. At zero error the bound returns `B>=mD`.

### Finite-length query-summary upper bound

This is an information benchmark using a public codebook, not a practical canonical lossy repair codec. Fix `0<v<1/4`, put `q=q(v)`, `r=floor(qn)`, `q_n=r/n`, and

\[
 V(n,r)=\sum_{j=0}^{r}\binom nj,\qquad
 K_{n,r}=\min\left\{2^n,
       \left\lceil\frac{2^n}{V(n,r)}(n\ln2+1)\right\rceil\right\}.
\]

There exists a Hamming covering code of radius `r` and size at most `K_(n,r)`. Taking the displayed number of independent uniform codewords gives expected uncovered population below one; taking all `2^n` words supplies the alternative bound. Fix a code publicly and a deterministic nearest-codeword rule.

Independently of the signs, draw a public uniform `n`-bit mask and uniform permutation. Mask and permute the private word, store only its covering-codeword index, and reconstruct each transformed coordinate as `q_n+(1-2q_n)*codeword_bit`. Undo the mask/permutation. For every transformed input, its coordinate-average squared error is at most

\[
 q_n^2+(1-2q_n)\frac{d_H(\text{input},\text{codeword})}{n}
             \le q_n(1-q_n)\le v.
\]

For any fixed original word and original coordinate, the uniform mask makes the transformed word independent of the permutation, and the permutation makes its queried coordinate uniform. Therefore each fixed original coordinate has expected squared error at most `v`. Returning the public head component plus the corresponding reconstructed private block gives, for every fixed input and every fixed hard probe,

\[
 \mathbb E\|\widehat\theta_i-\theta_i\|_2^2
                     \le4\alpha^2 v/t^2,
 \qquad B\le\lceil\log_2K_{n,r}\rceil.                       \tag{10}
\]

Return zero on non-hard requests. At `v=0` store exactly the signs; at `v=1/4` use the zero-information public-component predictor.

The usual type bound on `binom(n,r)` makes (10) equal to `n R_sq(v)+O_v(log(n+1))` in the standard finite-length sense for fixed interior `v`. Formula (10), not an unbounded-format limiting statement, is the authoritative bound. Codebook search and public mask/permutation storage can be enormous and are not charged as private information. Their physical cost remains real. Correctness is for each fixed query averaged over public randomness, not for a query chosen after seeing the realized state/coins. No efficient, canonical, or sequentially erasable lossy upper bound is asserted.

## 7. High-probability and certified numerical releases

A distribution-free success-probability statement also follows. Suppose each fixed hard probe has `||error||_2<=eta` with probability at least `1-delta`, for `delta<1/2`. A wrong sign in an `H` coordinate costs at least `alpha^2/(t^2 D)` squared error. Thus sign-threshold decoding has average bit-error rate at most

\[
 p=\min\left\{\tfrac12,\ \delta+(1-\delta)
                        \min\{1,t^2\eta^2/\alpha^2\}\right\},
\]

where one may replace the threshold rule by its Bayes-optimal bit decoder to obtain the cap `1/2`. Binary Fano and concavity yield `B>=n[1-h_2(p)]`. This conservative bound is separate from the sharper expected squared-error curve.

More directly, put

\[
                         a_H=\alpha/(t\sqrt D).
\]

If the successful-release radius is strictly less than `a_H`, **all** `D` private signs decode on a successful probe. Block Fano then gives

\[
 B\ge m\bigl[D-h_2(\delta)-\delta\log_2(2^D-1)\bigr]
      \ge m\bigl[(1-\delta)D-h_2(\delta)\bigr].              \tag{11}
\]

For deterministic certified releases, (11) is the full `mD`-bit lower bound even though the returned real vector need not equal the exact head.

As a concrete sufficient condition, if `0<lambda<=1` and the admitted dimension cap holds, then `t<4` and `sqrt(D)<=2^10`. Hence `a_H>2^-19`, and a rigorously certified parameter error at most `2^-20` suffices uniformly. This does not state that the repository's floating solver or residual-only gate has already established that error. Exact moment error, solve error, and release rounding must all be covered by the certificate used.

Without such fidelity conditions, exact-head distinguishability cannot simply be transferred to rounded outputs. Very large regularization can make distinct exact hidden coordinates underflow in a low-precision release. Numerical zero and raw signed-zero bit patterns are different output contracts; neither should be silently substituted for the exact rational target.

## 8. Interpretation and limits

The main strengthening is a concrete **finite-format witness**: normal FP32 raw inputs, public binary labels, the actual frozen cosine scorer, private information absent from the original head, and exact bounded-horizon ridge queries that identify that information. The conditional retained-state statement explains why an exact codec may discard a word only when it leaves the future admission family.

The information arguments themselves are classical. The squared-Bernoulli rate function and covering-code symmetrization specialize the repository's `round3_lower.md` argument to `mD` private feature bits; they are not a new general rate-distortion theorem. The fixed-format construction and scorer transfer are the additional qualifications. Broad bounded-deletion memory, provenance-polynomial maintenance, ridge add/delete maintenance, and finite-state coding all have prior literature.

Tadakamalla, Surisetty and Asad (2026), *Predictive State Is Not Explanatory State*, also study query-relative predictive/counterfactual memory. Their [official pinned release](https://github.com/PranayTadakamalla/predictive-state-not-explanatory-state/tree/4143f408843394e21a0622c9eb5d5de10a687a8c) implements reconstruction from all coordinate-flip threshold answers. We do not claim the general separation or partition-based sufficiency principle. The full paper was inaccessible in the current literature pass, leaving its approximate-success semantics unresolved; our precise fidelity contract is stated independently.

No claim follows about total-byte optimality, an efficient optimal lossy codec, uniform native runtime, privacy erasure, or genuine NLP utility. The same raw feature values feed graph normalization and ridge learning, but the rows are not unit learner features and are not an actual semantic-encoder image.

The loss in (8) is squared parameter norm. For example, a public test distribution uniformly choosing one `H` basis vector has prediction discrepancy `||P_H error||_2^2/D`; it dilutes the private-block effect by `D`. Isotropic unit-feature testing over all coordinates introduces the analogous `1/d` factor. Constant parameter-fidelity obstruction therefore does not establish constant task loss, population excess risk, or an observed semantic effect.

All source and numerical guarantees retain their declared access assumptions. In particular, public metadata may not contain the actual private candidate rows, and past released heads that a service can reuse are private-information side channels unless counted. The exact codec's public deletion ledger and serialization overhead are separate from its optimal private payload.

## 9. Verification and primary foundations

The companion [SCORER_TRANSFER.md](SCORER_TRANSFER.md) and `check_scorer_transfer.py` establish the declared uniform arithmetic envelope and bounded exact-FP32 examples; their frozen final report is `results/scorer_transfer_final/checks.json` (85 checks). The independent proof review checks algebra, information assumptions, and release distinctions. The separately owned canonical codec checker verifies actual sequential deletion and state encoding. These checks supplement the proofs; bounded examples do not enumerate the general `2^(mD)` family or establish the universal entropy inequalities by testing.

Primary foundations include Claude Shannon, *Coding Theorems for a Discrete Source With a Fidelity Criterion* (1959), [original-paper scan](https://gwern.net/doc/cs/algorithm/information/1959-shannon.pdf), and Jonathan Scarlett and Volkan Cevher, *An Introductory Guide to Fano's Inequality with Applications in Statistical Estimation*, arXiv:1901.00555v3, Theorem 1 and Sections 2.3 and 5.2: <https://arxiv.org/pdf/1901.00555>. Shannon's displayed binary Hamming curve is not itself the squared-soft-bit formula derived here; general reproduction alphabets and fidelity criteria supply the classical framework. Broad bounded-deletion storage remains prior work, including Cherapanamjeri et al., COLT 2025: <https://proceedings.mlr.press/v291/cherapanamjeri25b.html>. The novel contribution, if sustained by the overall literature comparison, must be the qualified curation-induced finite-word state phenomenon and its algorithmic consequences, not these tools.
