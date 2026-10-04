# Third-pass lower bounds: fixed-distribution prediction and a tight distortion curve

This note adds two proved results to the existing activation-frontier family. First, its constant-error storage obstruction is visible on a **single fixed public test distribution**, independent of the query and hidden labels. Second, the finite-bit obstruction has a matching asymptotic information-theoretic coding upper bound, both for failure probability and for expected squared error. These are model-query-summary results. Their upper bounds do **not** establish canonical deletion-clean state repair or an efficient practical codec.

## 1. Existing family and adversarial audit

Keep the second-pass construction unchanged. There are an anchor `a`, `b >= 1` common blockers `S`, private blockers `p_i`, and hidden records `w_i`, for `i=1,...,m`. Only the hidden labels `Y_i in {0,1}` vary. The priority order puts the anchor first, blockers next, hidden records last. The blocker sets satisfy `B(w_i)=S union {p_i}`. At cumulative budget `b` every retraining target is zero. At budget `b+1`, the request `F_i=S union {p_i}` selects exactly `{a,w_i}`.

The same unit embeddings are used for cosine curation and full-vector ridge learning:

    a=e_0,
    s_j=(1/2)e_0+(sqrt(3)/2)e_1,
    p_i=(1/2)e_0+(sqrt(3)/2)u_i,
    w_i=(1/2)e_1+(sqrt(3)/2)u_i,

where all signature vectors are orthogonal to `e_0,e_1`, unit length, and satisfy `|u_i dot u_j|<=1/6`. Strict threshold `2/5` realizes the required graph. There are such signatures in dimension `O(log m)`. Let `c=1+2 lambda`, where `lambda>0`. The retained objective and optimum at the hard request are

    L_i(theta) = (1/4)[(a dot theta)^2+(w_i dot theta-Y_i)^2]
                 +(lambda/2)||theta||^2,
    theta_i = Y_i w_i/c.

All hard-request payloads are public and independent of `Y`. All hidden-label-dependent retained information counts in the `B`-bit state. Public randomness is independent of `Y`; codebooks and embeddings are public. There is no retained-label reread service or free dataset-dependent ticket.

The existing proof survives the following checks:

- At low budget, deletion of the anchor can admit zero-label blockers but cannot admit a hidden record: each hidden record has `b+1` blockers. Its learner is still zero.
- At high budget, `F_i` is the **only** type of request of size at most `b+1` that selects a hidden record. Any such selection must delete its exact `b+1` blockers, leaving no budget for anything else.
- Private-blocker and candidate cross-correlations are separated from the strict threshold by a constant. The logarithmic-dimension existence argument uses independent labels, not a claim about natural text.
- The binary-entropy lower bound requires no simultaneous-correctness event over all indices.
- The fixed bit bound must count information in model parameters and caches, and exclude uncounted retained-label access. Treating arbitrary exact real numbers as single finite-bit words would invalidate a bit interpretation.

## 2. A fixed public test distribution witnesses the obstruction

For a test-feature distribution `mu`, write `||f||_mu^2=E_{X~mu} f(X)^2`. Define the known unit-label oracle function

    f_i(x) = x dot w_i/c.

The target predictor at request `F_i` is `Y_i f_i`. The repair may output an arbitrary measurable predictor, not necessarily a linear parameter vector.

**Theorem P (fixed-distribution prediction lower bound).** Suppose a public test distribution `mu`, independent of `Y` and the query, satisfies

    d_* := min_i ||f_i||_mu > 0.

If, for every fixed label vector and hard request, repair has

    Pr(||f_hat_i-Y_i f_i||_mu <= eta) >= 1-delta,
    eta < d_*/2,  delta<1/2,

then

    B >= m[1-h_2(delta)].

**Proof.** The two possible targets for query `i` are the public functions zero and `f_i`, separated by `||f_i||_mu`. Choose whichever is closer to the repaired function. The triangle inequality implies this nearest-template decision is correct on the stated accuracy event. Apply the same conditional binary-entropy proof as in the existing finite-bit theorem. Fresh private decoder randomness cannot beat the Bayes-optimal decoder. QED.

This is not merely a query-specific test point: the following choices use one distribution for all hard requests.

### 2.1 Public candidate-feature test distribution

Let `mu` be uniform on the `m` public features `w_1,...,w_m`. Every point in this support survives every hard request. For `j!=i`,

    w_j dot w_i = 1/4+(3/4)(u_j dot u_i) >= 1/8,

while `w_i dot w_i=1`. Consequently

    ||f_i||_mu^2 >= [1+(m-1)/64]/(m c^2) >= 1/(64c^2).

Thus ordinary root-mean-square prediction error `eta < 1/(16c)` implies the same linear-bit obstruction. This is a fixed distribution over public retained-candidate feature locations; it does not require access to their hidden labels.

### 2.2 A full-rank unit-feature distribution with exactly equal target energy

Let the signatures live in `R^d` with orthonormal basis `e_2,...,e_{d+1}`. Give `e_0` and `e_1` probability `1/4` each, and give each signature-basis vector probability `1/(2d)`. All test features are unit vectors. The distribution has full-rank second moment and is independent of the index and labels. For every `i`,

    ||f_i||_mu^2 = a_d^2 := [1/16+3/(8d)]/c^2 >= 1/(16c^2).

Hence `eta < a_d/2` suffices; the simpler dimension-free condition `eta < 1/(8c)` suffices. This variant yields the exact ordinary-MSE curve below, without query-dependent normalization. It is a constructed public test distribution, not a natural-language population claim.

### 2.3 Limits of the claim

Prediction fidelity to the retained-data oracle is not the same as excess risk on an independently specified true-label population. For example, if all fixed test labels are zero, the constant-zero predictor can have lower population risk than some retained-data ridge targets while failing to reproduce them. No universal positive lower bound follows from a one-sided condition `R(f_hat)-R(f_oracle)<=epsilon` on an arbitrary fixed supervised test law. Do not call Theorem P a population-excess-risk theorem.

## 3. Tight information rate for squared-error query fidelity

Let `D in [0,1/4]` and define

    q(D) = [1-sqrt(1-4D)]/2,
    R_sq(D) = 1-h_2(q(D)).

For `D>=1/4`, let `R_sq(D)=0`.

**Theorem Q (sharp asymptotic model-query information curve).** In the activation-frontier family, require expected normalized prediction error

    E ||f_hat_i-Y_i f_i||_mu^2 / ||f_i||_mu^2 <= D

for every fixed `Y,i`; averaging over independent uniform `Y,i` would already suffice for the lower bound. Then every finite-bit summary satisfies

    B >= m R_sq(D).

Conversely, for each fixed `D in (0,1/4)`, there are public randomized model-query summaries with

    B <= m R_sq(D)+O(log m)

and expected normalized error at most `D` for every fixed `Y,i`. The endpoint `D=0` uses exactly `m` bits; `D=1/4` uses zero hidden-label-dependent bits by returning `f_i/2`. Non-hard requests return their exact zero model. The upper bound concerns per-fixed-query correctness averaged over public randomness. It does not guarantee correctness against an adversary that chooses a query after seeing the realized randomness/state.

### 3.1 Converse proof, including the entropy inequality

Project an arbitrary predicted function onto the known span of `f_i`:

    Y_hat_i = <f_hat_i,f_i>_mu / ||f_i||_mu^2.

Cauchy-Schwarz shows `(Y_hat_i-Y_i)^2` is at most the normalized function error. Clipping to `[0,1]` cannot increase squared error. Let `R` be public randomness and `p_i=Pr(Y_i=1|M,R)`, for independent uniform hidden bits `Y`. The minimum conditional squared error is `p_i(1-p_i)`; therefore

    (1/m) sum_i E p_i(1-p_i) <= D.

Set `g(v)=h_2([1-sqrt(1-4v)]/2)` for `v in [0,1/4]`. It is increasing and concave. To verify concavity, write `v=q(1-q)` for `q<=1/2`; the sign of the derivative of `g'(v)` is that of

    -(1-2q)/(q(1-q)) + 2 ln((1-q)/q).

With `t=(1-q)/q>=1`, this is `-t+1/t+2 ln t<=0`, because `t-1/t-2 ln t` is zero at one and has derivative `(t-1)^2/t^2>=0`. Continuity covers the endpoints. Since `h_2(p)=g(p(1-p))`, Jensen gives

    H(Y|M,R) <= sum_i E h_2(p_i)
              <= m g((1/m)sum_i E p_i(1-p_i))
              <= m g(D).

Finally `B>=I(Y;M|R)=m-H(Y|M,R)>=m[1-g(D)]`. For `D>=1/4` the zero lower bound is immediate. QED.

### 3.2 Achievability proof by covering codes

Fix `q=q(D) in (0,1/2)`, `r=floor(qm)`, `q_m=r/m`, and let

    V(m,r)=sum_{j=0}^r binom(m,j).

There exists a binary Hamming covering code `C subset {0,1}^m` of radius `r` with

    |C| <= ceil([2^m/V(m,r)](m ln 2+1)).

Indeed, choose that many independent uniform codewords. A fixed word is uncovered with probability at most `exp(-|C|V/2^m)`. The expected number of uncovered words is strictly below one, so some realized code covers all words. Fix such a code publicly and a deterministic nearest-codeword encoder.

Draw a public uniform mask `Z in {0,1}^m` and uniform permutation `pi`, independently of `Y`. Encode

    U=pi(Y xor Z)

by storing only the index of a covering codeword `C(U)`. At query `i`, put `j=pi(i)` and reconstruct the transformed coordinate as

    U_hat_j = q_m+(1-2q_m) C(U)_j.

Undo mask bit `Z_i` to obtain `Y_hat_i`. Squared error is unchanged by that unmasking. For every transformed word `U`, the coordinate-mean squared error is at most

    q_m^2+(1-2q_m)[d_H(U,C(U))/m]
    <= q_m(1-q_m) <= D.

For any fixed `Y,i`, the random vector `U` is uniform and independent of `pi`, while `pi(i)` is a uniform coordinate independent of `U`. Hence the expected coordinate error obeys the same bound for every fixed `Y,i`. Return `Y_hat_i f_i` for the hard request.

The code index takes at most

    ceil(log_2 |C|)
    <= m-log_2 V(m,r)+O(log m)
    <= m[1-h_2(q_m)]+O(log m)
    = m R_sq(D)+O(log m)

bits for fixed interior `D`. The penultimate step uses the elementary type bound `binom(m,r)>=2^{m h_2(r/m)}/(m+1)`. The mask, permutation, and codebook are public independent randomness/metadata, not hidden-label-dependent state. They still have physical storage and access costs in any implementation. QED.

This is classical rate-distortion coding specialized to the counterfactual-activation family, not a new general coding theorem. The code may be expensive to construct or search. It establishes tight information requirements, not a practical compressed repair implementation.

### 3.3 Ordinary MSE on one fixed distribution

Using the full-rank distribution in Section 2.2, all oracle template energies equal `a_d^2`. If the requested expected ordinary prediction MSE is `epsilon`, put `D=epsilon/a_d^2`. Thus the exact asymptotic label-information rate is

    B/m = R_sq(epsilon/a_d^2)+o(1).

Its zero-information threshold is `epsilon=a_d^2/4`, bounded below by `1/(64c^2)` independently of `m`. Every fixed strictly smaller normalized tolerance requires a positive linear information rate. The low-budget side still requires zero hidden-label-dependent bits for exact repair.

## 4. Tight retained-objective excess-risk curve

The ridge objective at the hard query gives an even simpler prediction-independent corollary.

**Corollary E.** If `E[L_i(theta_hat)-L_i(theta_i)]<=epsilon` for every fixed hidden label vector and hard query, then

    B >= m R_sq(4c epsilon).

Conversely the coded query summary attains expected excess objective at most `epsilon` using `m R_sq(4c epsilon)+O(log m)` hidden-label-dependent bits, for every fixed `0<epsilon<1/(16c)`. At `epsilon=1/(16c)` zero hidden-label information suffices; at zero error exact `m`-bit storage suffices.

**Proof.** The objective is quadratic with Hessian `H_i=lambda I+(aa^T+w_iw_i^T)/2`. The vectors `a,w_i` are orthonormal. Let `Y_hat_i=c(w_i dot theta_hat)`. Then

    L_i(theta_hat)-L_i(theta_i)
      = (1/2)(theta_hat-theta_i)^T H_i(theta_hat-theta_i)
      >= (Y_hat_i-Y_i)^2/(4c).

The lower bound follows from Theorem Q's scalar argument with `D=4c epsilon`. Equality holds if `theta_hat=Y_hat_i w_i/c`, so the coding construction attains the matching bound. At zero hidden-label memory choose `Y_hat_i=1/2`. Its excess objective is exactly `1/(16c)` for either bit. QED.

For a high-probability excess-objective guarantee with tolerance strictly below `1/(16c)`, nearest-bit decoding also gives the original `m[1-h_2(delta)]` lower bound. This is excess risk on the *counterfactual retained empirical regularized objective*, not on an arbitrary fixed supervised population.

## 5. The failure-probability bound is itself asymptotically tight

Use the same covering construction with radius `r=floor(delta m)`, but decode a binary codeword bit rather than the posterior-valued reproduction. Mask/permutation symmetrization gives bit error at most `delta` for every fixed `Y,i`. Return the exact model corresponding to that decoded bit. Thus for every fixed `delta in (0,1/2)` and any tolerance below half the template separation,

    m[1-h_2(delta)] <= B <= m[1-h_2(delta)]+O(log m)

under the model-query-summary/public-randomness contract. This closes the original finite-bit lower bound at its stated per-query success metric. It still does not yield a state-clean codec or an adaptive-query upper bound.

## 6. Recommended incorporation and claim boundaries

1. Add fixed-distribution prediction as a corollary immediately after the shared-embedding ridge construction. The candidate-feature distribution gives the clearest link to retained data.
2. Add the retained-objective excess-risk curve as the main quantitative sharpening; it has a simple constant `4(1+2lambda)` and a matching zero-information threshold.
3. Put the complete rate-distortion/coding proof in an appendix. State explicitly that it characterizes information in a query summary, not a practical canonical-state repair data structure.
4. Do not claim a universal population-risk lower bound, an efficient optimal codec, fixed-dimensional embeddings for arbitrarily large `m`, or privacy-safe deletion from the compressed code.
5. Treat the information inequalities/coding argument as standard tools. The research claim remains the curation-induced activation frontier and how it changes the information required by a specified learner.

Primary foundation for rate-distortion terminology: Claude E. Shannon, *Coding Theorems for a Discrete Source With a Fidelity Criterion*, IRE National Convention Record 7(4):142–163, 1959. Publisher record: https://ieeexplore.ieee.org/document/5311476 ; original primary-paper scan: https://www.mast.queensu.ca/~math474/shannon59.pdf . Retrieved search refs `turn54search2`, `turn54search24`; parent must open a source itself before citing it in a web-cited final response. The proof above is self-contained and does not rely on an uncited specialized formula.
