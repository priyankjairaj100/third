# Exact FP32 family under the frozen Phase 3 scorer

This note proves the graph claim for the actual ordered-coordinate scoring
source. It is a theorem about a finite algebraic family of stored vectors, not
semantic-embedding evidence or a new empirical dataset. No source in Phases
3–9 changes. The associated finite-word ridge theorem uses the original stored
features; curator normalization below is not a change of learner features.

## Statement and arithmetic contract

Let `k=4^r`, `D=4^s` for nonnegative integers `r,s`, and let
`d=2+k+D <= 2^20`. The four mutually orthogonal coordinate blocks are
`e0`, `e1`, `U` of length `k`, and `H` of length `D`. Choose public signatures
`u_i in {−1/sqrt(k),+1/sqrt(k)}^k` with
`|<u_i,u_j>| <= 1/6` for distinct indices. Independently choose **any** private
sign vectors `z_i in {−1/sqrt(D),+1/sqrt(D)}^D`. Put

\[
a=e_0,\quad s_j=\tfrac12e_0+\tfrac78e_1,\quad
p_i=\tfrac12e_0+\tfrac78u_i,\quad
w_i=\tfrac12e_1+\tfrac78u_i+\tfrac1{128}z_i.
\]

There may be any finite number `b` of common blockers `s_j`, and any finite
signature codebook satisfying the stated bound. All coordinates above are
stored directly in binary32, without preliminary unit normalization. The
threshold is the binary64 value `tau=float(0.4)`. Assign public identifiers so
their fixed priority order is `a`, the `s_j`, the `p_i`, then the `w_i`.

Assume the source is executed with ordinary IEEE binary32/binary64 formats,
binary64 round-to-nearest ties-to-even operations, correctly rounded square
root and division, and no arithmetic-changing fast-math replacement. The
conversion binary32 to binary64 is exact. This is an explicit execution
contract, not a universal assertion about every NumPy build or hardware mode.
The range argument below excludes overflow and nonzero subnormal arithmetic
for this family. Empirical probes of a machine cannot prove its compliance for
all possible inputs.

**Scorer-transfer theorem.** Under this contract, the frozen functions
`phase3.panels.graph_normalize`, `phase3.panels.reference_cosines`, and
`phase3.reference_graph.build_reference_graph` produce, for every private sign
assignment, exactly the intended strict-threshold graph. Every edge score is
greater than `tau+29/1000`; every distinct nonedge score is less than
`tau−27/1000`. The blocker sets are

\[
B(a)=\varnothing,\quad
B(s_j)=\{a,s_1,\ldots,s_{j-1}\},\quad
B(p_i)=\{a\},\quad
B(w_i)=\{s_1,\ldots,s_b,p_i\}.
\]

The same assertion holds after any deletions, with blocker sets restricted to
surviving records and their original relative priorities. No assertion about
natural text encoders or primary threshold calibration follows.

## Exact geometry, uniformly over the private signs

The squared norms are

\[
\|a\|^2=1,\quad \|s_j\|^2=\|p_i\|^2=65/64,\quad
\|w_i\|^2=16641/16384=(129/128)^2.
\]

Write `t=<u_i,u_j>` and `h=<z_i,z_j>`. For distinct signatures,
`|t|<=1/6`; for **all** private choices, `|h|<=1` by Cauchy–Schwarz.
The complete distinct-record pair table is:

| Pair | Exact normalized cosine or upper bound | Edge? |
| --- | --- | --- |
| `a,s_j` or `a,p_i` | `4/sqrt(65)` | Yes |
| `a,w_i` | `0` | No |
| Different common blockers | `1` | Yes |
| `s_j,p_i` | `16/65` | No |
| `s_j,w_i` | `448/(129 sqrt(65))` | Yes |
| `p_i,w_i` | `784/(129 sqrt(65))` | Yes |
| `p_i,p_j`, `i!=j` | `(16+49t)/65 <= 29/78` | No |
| `p_i,w_j`, `i!=j` | `784t/(129 sqrt(65)) <= 392/(387 sqrt(65))` | No |
| `w_i,w_j`, `i!=j` | `(4096+12544t+h)/16641 <= 18563/49923` | No |

The smallest edge cosine is `448/(129 sqrt(65)) > 43/100`.
The largest nonedge upper bound is `18563/49923 < 93/250`.
For radical entries these inequalities follow by squaring positive quantities;
all remaining comparisons are rational cross-products. In particular, no
packing or sampling assumption on the private signs is needed.

## Exact input representation and norm accumulation

Since `sqrt(k)=2^r` and `sqrt(D)=2^s`, every nonzero coordinate is among
`1`, `1/2`, `7/8`, `±7*2^(-r-3)`, and `±2^(-s-7)`.
The dimension bound implies `r,s<=10` (a deliberately loose bound), so these
are normal binary32 numbers with at most three nonzero significand bits.
There is no quantization error. The smallest possible nonzero magnitude under
this loose bound is `2^-17`.

The frozen normalizer first promotes values to FP64, then, in coordinate
order, performs the square and adds it into an FP64 accumulator. The possible
nonzero squares are `1`, `1/4`, `49/64`, `49*2^(-2r-6)`, and
`2^(-2s-14)`. Each square is exactly representable in FP64. Every norm partial
sum is a nonnegative integer multiple of `2^-34`, and is less than `2`.
Its numerator on that common grid is therefore below `2^35`, far below
binary64's 53-bit significand capacity. By induction each multiply and each
ordered addition is exact, including all intervening zero coordinates.
This does not depend on the signs or number of records in a tile.

Consequently the computed squared norms equal the displayed rational values
exactly. The square roots for `a` and `w_i` are exactly `1` and `129/128`.
Only the square root `sqrt(65)/8` for common/private blockers needs rounding.
The proof below nevertheless permits one square-root error for every row,
which gives a single uniform bound.

## Bound every remaining FP64 operation

Let `u=2^-53` be binary64 unit roundoff. For a stored row `x`, let
`v=x/||x||` denote its exact unit normalization and `vhat` the frozen computed
normalization. A nonzero coordinate satisfies

\[
\widehat v_\ell=v_\ell\frac{1+\delta_\ell}{1+\delta_s},
\qquad |\delta_\ell|,|\delta_s|\le u.
\]

Here `delta_s` accounts for square root and `delta_l` for the coordinate
division. Zero coordinates remain exactly zero. Define

\[
\rho=\frac{2u}{1-u}.
\]

Then `|vhat_l-v_l| <= rho |v_l|`. For another normalized row `w`,
Cauchy–Schwarz gives

\[
\left|\sum_\ell\widehat v_\ell\widehat w_\ell-
\sum_\ell v_\ell w_\ell\right|\le 2\rho+\rho^2,
\qquad
\sum_\ell|\widehat v_\ell\widehat w_\ell|\le(1+\rho)^2.
\]

`reference_cosines` separately rounds each product and each addition, in the
same coordinate order, starting from zero. Set

\[
\gamma_n=\frac{nu}{1-nu}.
\]

Expanding the sequential recurrence, each product term acquires at most `d+1`
factors of the form `1+delta`, `|delta|<=u`. The deviation of any such product
from one is at most `gamma_(d+1)`: the upper side follows from
`(1+u)^n <= 1/(1-nu)` for `nu<1`, and the lower side from
`(1-u)^n >= 1-nu`. Applying the triangle inequality to the absolute terms
therefore proves the conservative ordered-dot bound

\[
|\widehat c-c|\le E_d:=2\rho+\rho^2+
\gamma_{d+1}(1+\rho)^2.\tag{1}
\]

This absolute bound remains valid with severe cancellation; it does not divide
by the exact dot product. It includes multiplication, accumulation, square
root, and division errors. There is no BLAS reduction or corpus-size-dependent
sum in the frozen path.

The function `E_d` is increasing in `d` on the permitted range. Substitution of
`d=2^20` into (1), using only rational arithmetic, gives

\[
E_d\le E_{2^{20}}<2^{-32}.
\]

For a simple analytic comparison, `rho<3u`,
`(2^20+1)u < (9/8)2^-33`, and
`1/(1-(2^20+1)u)<8/7`. Thus `gamma_(d+1)<(9/7)2^-33`,
`(1+rho)^2<9/8`, and `2rho+rho^2<7u`; together these give
`E_d < (81/56)2^-33 + 7*2^-53 < 2*2^-33`.
The retained checker also verifies the sharper exact rational expression.

There is no hidden underflow exception in this argument. Computed nonzero
normalized coordinates lie between `2^-18` and `2` in magnitude, so a nonzero
rounded product lies between `2^-37` and `4`. All these products are binary64
multiples of `2^-89`. By induction each accumulated result is also a multiple
of that grid: exact cancellation gives zero, and a nonzero value is at least
`2^-89`. Rounding a grid value either leaves it exact at finer spacing or
rounds to a coarser subgrid. All values are vastly above the binary64 subnormal
range. Even the crude intermediate magnitude bound `4d <= 2^22` excludes
overflow. Norm squares, square roots, and coordinate divisions have already
been bounded away from zero and infinity.

## Stored threshold, strict comparison, and absence of clipping

The exact value of the Python binary64 threshold is

\[
\tau=\frac{3602879701896397}{9007199254740992}
=\frac25+\frac1{5\,2^{53}}.
\]

Combining the rational geometry margins with (1),

\[
\widehat c_{\rm edge}>43/100-2^{-32}>\tau+29/1000,
\qquad
\widehat c_{\rm nonedge}<93/250+2^{-32}<\tau-27/1000.
\]

Thus the source's actual comparison `scores > threshold` has no equality or
rounding ambiguity. FP64 comparison itself is exact.

**The frozen scorer contains no clipping operation.** `reference_cosines`
returns its ordered sums directly. A hypothetical extra projection onto
`[-1,1]` would be nonexpansive relative to the exact cosine and would preserve
the strict decision at this interior threshold, but it is not used in this
source-bound theorem. Identical common-blocker rows may have a computed score
slightly above one; this does not affect their edge decision.

## Public priority assignment and restriction

Take any requisite number of distinct public IDs and any fixed nonnegative
seed. Evaluate the actual `stable_priority` function once; assign the anchor,
common-blocker, private-blocker, and candidate roles to successive IDs in that
sorted order. This assigns payload roles **after sorting**; it does not seek
IDs with prescribed SHA-256 outputs. Ties already use the public ID as the
secondary key. The codebook, role assignment, IDs and seed are public and
independent of all private `z_i`.

On deletion, sorting the surviving subset with the same total key preserves
relative order. Normalization and pair scoring use the same original stored
rows, coordinate order and arithmetic. The preceding uniform strict margins
therefore give precisely the induced surviving graph, independent of tile
size. The blocker-set statement follows directly from the pair table.

## Source binding and bounded checks

The scorer sources proved here are:

| Source | SHA-256 |
| --- | --- |
| `phase3/panels.py` | `bd2409958ac9a9e27faf463b44b2f839260a3195c7cde32fb3bfb76aeb14a04b` |
| `phase3/reference_graph.py` | `efc41498648f6f7f7969768fbcd050619d7ed5d6e82eafadb1b9a3a2158e9d1c` |
| `ccu/core.py` | `d89f6affd566103c65aa92f10e02ac86e394f7a1f88ff2e6b098720b8cef7df3` |

`check_scorer_transfer.py` refuses a changed frozen source. It checks the exact
rational constants and every admissible exponent pair without constructing
large arrays, then exercises the actual graph functions on eight small
algebraic payloads at three tile sizes and on a retained subset. These checks
support the source/formula connection. The analytic argument, not enumeration
of those payloads, proves the claim for all private assignments and dimensions.
The checker is not an empirical benchmark, hardware conformance certificate,
or primary-data intake. It performs no model training or annotation.

```bash
python3 empirical_execution/phase10/check_scorer_transfer.py \
  --out empirical_execution/phase10/results/scorer_transfer_final
```

Existing outputs are never overwritten. The report binds the proof, checker,
frozen sources, runtime description and bounded outcomes.

## Arithmetic references

The [Oracle Numerical Computation Guide, IEEE Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_math.html)
states the 24/53-bit significand formats, nearest-even mode, and correct
rounding requirements for arithmetic including square root and division.
The [official NumPy scalar documentation](https://numpy.org/doc/stable/reference/arrays.scalars.html)
identifies `float32` and `float64`; the [sqrt documentation](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html)
identifies that elementwise operation. These references specify the arithmetic
model and interfaces, not empirical certification of an arbitrary build.
Retrieved 4 October 2026. The roundoff inequalities above are derived here.
