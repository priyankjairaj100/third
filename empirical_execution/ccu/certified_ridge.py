"""Exact residual certificates for an approximate regularized ridge solution.

This module treats every supported finite float (up to binary64) as its exact
dyadic rational. Extended NumPy floats are rejected; explicit Fraction inputs
remain available for an exact larger-precision target.
It does not trust the candidate numerical solver and does not round a residual
or an error bound.  A certificate binds to the exact matrices supplied here;
using rounded factors certifies those factors, not an unrecorded raw-data target.

For n > 0, W* solves (G + lambda*n*I) W* = H.  After proving G is positive
semidefinite in rational arithmetic, the certificate is

    ||W-W*||_F^2 <= ||(G+lambda*n*I) W-H||_F^2 / (lambda*n)^2.

The factor interface evaluates G=U A U.T and H=U B without forming an ambient
Gram matrix.  U need not be orthonormal.  Positive semidefiniteness of A is a
sufficient, explicitly checked condition.  It can be unnecessarily strict if
U is column-rank deficient; such inputs are rejected, never falsely certified.
Empty training sets use the specified zero-head convention and require zero
encoded moments.  Their error is computed directly, without dividing by n.

Exact fractions can grow large: this is an optional verification layer, not
part of a constant-time or bounded-bit-size numerical repair claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from numbers import Integral
from typing import Iterable, Sequence


class CertificateInputError(ValueError):
    """The exact target failed an explicit certificate precondition."""


def rational(value) -> Fraction:
    """Convert a scalar exactly; finite floats preserve their binary value."""
    if isinstance(value, bool):
        raise CertificateInputError("Boolean values are not numeric target data")
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CertificateInputError("Non-finite floating point input")
        return Fraction.from_float(value)
    # The persisted numerical contract supports at most binary64. NumPy
    # extended floats are deliberately rejected, even when their platform
    # implementation exposes an exact ratio. Never recurse through item().
    dtype = getattr(value, "dtype", None)
    if getattr(dtype, "kind", None) == "f" and getattr(dtype, "itemsize", 0) > 8:
        raise CertificateInputError("Extended floating-point types are unsupported; supply an explicit exact Fraction")
    # Decimal and supported NumPy values can expose exact ratios. Avoid
    # narrowing them through Python float (or recursively calling item()).
    if hasattr(value, "as_integer_ratio"):
        try:
            numerator, denominator = value.as_integer_ratio()
            return Fraction(int(numerator), int(denominator))
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise CertificateInputError("Non-finite or invalid exact-ratio input") from exc
    # NumPy scalars provide item(); conversion of float32 to Python float is exact.
    if hasattr(value, "item"):
        converted = value.item()
        if type(converted) is not type(value):
            return rational(converted)
    try:
        return Fraction(value)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as exc:
        # SymPy-compatible exact rationals commonly expose p and q.
        if hasattr(value, "p") and hasattr(value, "q"):
            try:
                return Fraction(int(value.p), int(value.q))
            except Exception:
                pass
        raise CertificateInputError(f"Unsupported exact scalar type {type(value).__name__}") from exc


def _matrix(values, name: str, rows: int | None = None,
            columns: int | None = None) -> list[list[Fraction]]:
    try:
        out = [[rational(value) for value in row] for row in values]
    except TypeError as exc:
        raise CertificateInputError(f"{name} must be a rectangular matrix") from exc
    if rows is not None and len(out) != rows:
        raise CertificateInputError(f"{name} has wrong row count")
    width = columns if columns is not None else (len(out[0]) if out else 0)
    if any(len(row) != width for row in out):
        raise CertificateInputError(f"{name} has wrong or inconsistent column count")
    return out


def _count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise CertificateInputError("count must be a nonnegative integer")
    return int(value)


def _prove_psd(matrix: list[list[Fraction]], name: str) -> int:
    """Exact unpivoted Schur-complement PSD proof, including singular pivots.

    For a PSD matrix, a zero diagonal entails an identically zero corresponding
    row/column.  A negative pivot or nonzero zero-pivot row disproves PSD.
    Every successful positive pivot leaves a PSD iff original Schur complement.
    """
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise CertificateInputError(f"{name} must be square")
    for i in range(size):
        for j in range(i):
            if matrix[i][j] != matrix[j][i]:
                raise CertificateInputError(f"{name} is not exactly symmetric")
    work = [row[:] for row in matrix]
    rank = 0
    for k in range(size):
        pivot = work[k][k]
        if pivot < 0:
            raise CertificateInputError(f"{name} is not positive semidefinite (negative pivot {k})")
        if not pivot:
            if any(work[j][k] for j in range(k + 1, size)):
                raise CertificateInputError(f"{name} is not positive semidefinite (zero pivot row {k})")
            continue
        rank += 1
        column = [work[i][k] for i in range(k + 1, size)]
        for ii, i in enumerate(range(k + 1, size)):
            if not column[ii]:
                continue
            factor = column[ii] / pivot
            for jj in range(ii, len(column)):
                j = k + 1 + jj
                if column[jj]:
                    work[j][i] -= factor * column[jj]
                    work[i][j] = work[j][i]
    return rank


def _matmul(left, right, right_columns: int | None = None):
    width = len(right[0]) if right else (right_columns or 0)
    if any(len(row) != len(right) for row in left):
        raise CertificateInputError("Matrix multiplication shape mismatch")
    out = [[Fraction(0) for _ in range(width)] for _ in left]
    for i, row in enumerate(left):
        for k, value in enumerate(row):
            if value:
                for j, coefficient in enumerate(right[k]):
                    if coefficient:
                        out[i][j] += value * coefficient
    return out


def _transpose(matrix, columns=None):
    if matrix:
        return [list(column) for column in zip(*matrix)]
    return [[] for _ in range(columns or 0)]


def _squared_frobenius(matrix) -> Fraction:
    return sum((entry * entry for row in matrix for entry in row), Fraction(0))


def _fraction_json(value: Fraction):
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def sqrt_upper_rational(squared, decimal_places: int = 30) -> Fraction:
    """ceil(10**p * sqrt(squared))/10**p, calculated with integers only."""
    value = rational(squared)
    if value < 0:
        raise CertificateInputError("Squared bound must be nonnegative")
    if isinstance(decimal_places, bool) or not isinstance(decimal_places, Integral) or not 0 <= decimal_places <= 10000:
        raise CertificateInputError("decimal_places must be an integer between 0 and 10000")
    scale = 10 ** int(decimal_places)
    numerator = value.numerator * scale * scale
    denominator = value.denominator
    root = math.isqrt(numerator // denominator)
    if root * root * denominator < numerator:
        root += 1
    return Fraction(root, scale)


def _fixed_decimal(upper: Fraction, places: int) -> str:
    scale = 10 ** places
    scaled = upper * scale
    if scaled.denominator != 1:
        raise ArithmeticError("Upper-bound denominator is not a decimal divisor")
    whole, decimals = divmod(scaled.numerator, scale)
    return str(whole) if not places else f"{whole}.{decimals:0{places}d}"


def _target_digest(kind, matrices, count, lam):
    digest = hashlib.sha256()
    digest.update(f"certified-ridge-v1:{kind}:{count}:{lam.numerator}/{lam.denominator}\n".encode())
    for name, matrix in matrices:
        digest.update(f"{name}:{len(matrix)}:{len(matrix[0]) if matrix else 0}\n".encode())
        for row in matrix:
            for value in row:
                digest.update(f"{value.numerator}/{value.denominator};".encode())
            digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True)
class RidgeCertificate:
    target_digest: str
    candidate_digest: str
    count: int
    feature_dimension: int
    response_dimension: int
    lambda_reg: Fraction
    residual_squared_frobenius: Fraction
    error_squared_upper_bound: Fraction
    psd_core_rank: int
    target_representation: str
    exact_arithmetic: str = "Python Fraction; finite floats interpreted as exact dyadic rationals"
    empty_target: bool = False

    def upper_bound(self, decimal_places: int = 30) -> Fraction:
        return sqrt_upper_rational(self.error_squared_upper_bound, decimal_places)

    def meets_tolerance(self, tolerance) -> bool:
        tolerance = rational(tolerance)
        if tolerance < 0:
            raise CertificateInputError("tolerance must be nonnegative")
        return self.error_squared_upper_bound <= tolerance * tolerance

    def prediction_squared_upper_bound(self, feature_row) -> Fraction:
        row = [rational(value) for value in feature_row]
        if len(row) != self.feature_dimension:
            raise CertificateInputError("feature_row dimension mismatch")
        return sum((value * value for value in row), Fraction(0)) * self.error_squared_upper_bound

    def to_dict(self, decimal_places: int = 30):
        upper = self.upper_bound(decimal_places)
        return {
            "schema": "exact-ridge-residual-certificate-v1",
            "target_digest": self.target_digest,
            "candidate_digest": self.candidate_digest,
            "count": self.count,
            "feature_dimension": self.feature_dimension,
            "response_dimension": self.response_dimension,
            "lambda_reg_exact": _fraction_json(self.lambda_reg),
            "residual_squared_frobenius_exact": _fraction_json(self.residual_squared_frobenius),
            "error_squared_upper_bound_exact": _fraction_json(self.error_squared_upper_bound),
            "error_upper_bound_rational": _fraction_json(upper),
            "error_upper_bound_decimal": _fixed_decimal(upper, decimal_places),
            "psd_core_rank": self.psd_core_rank,
            "target_representation": self.target_representation,
            "exact_arithmetic": self.exact_arithmetic,
            "empty_target": self.empty_target,
            "bound_semantics": "Frobenius distance from candidate W to the exact specified ridge minimizer; no distributional unlearning claim",
        }


def _finish(kind, matrices, count, lam, weights, residual, rank, empty=False):
    squared = Fraction(0) if empty else _squared_frobenius(residual)
    bound = _squared_frobenius(weights) if empty else squared / (lam * count) ** 2
    candidate_digest = _target_digest("candidate", [("weights", weights)], count, lam)
    return RidgeCertificate(_target_digest(kind, matrices, count, lam), candidate_digest, count,
                            len(weights), len(weights[0]) if weights else 0,
                            lam, squared, bound, rank, kind, empty_target=empty)


def exact_residual_certificate(gram, cross, count, lambda_reg, weights) -> RidgeCertificate:
    """Prove a rigorous ridge error bound for a supplied dense exact target."""
    count, lam = _count(count), rational(lambda_reg)
    if lam <= 0:
        raise CertificateInputError("lambda_reg must be strictly positive")
    weights = _matrix(weights, "weights")
    d, c = len(weights), len(weights[0]) if weights else 0
    if d < 1 or c < 1:
        raise CertificateInputError("weights must have positive dimensions")
    gram = _matrix(gram, "gram", d, d)
    cross = _matrix(cross, "cross", d, c)
    matrices = [("gram", gram), ("cross", cross)]
    if not count:
        if any(value for row in gram + cross for value in row):
            raise CertificateInputError("An empty training target must have exactly zero moments")
        return _finish("dense", matrices, count, lam, weights, weights, 0, empty=True)
    rank = _prove_psd(gram, "gram")
    residual = _matmul(gram, weights)
    for i in range(d):
        for j in range(c):
            residual[i][j] += lam * count * weights[i][j] - cross[i][j]
    return _finish("dense", matrices, count, lam, weights, residual, rank)


def factor_residual_certificate(basis, gram_core, cross_core, count,
                                lambda_reg, weights) -> RidgeCertificate:
    """Certify G=U A U.T, H=U B without forming G or requiring U.T U=I.

    Exact symmetry and PSD of A are checked.  A successful check proves PSD of
    the represented ambient G even when U is not orthonormal/full rank.  For an
    empty target, PSD(A) plus U A U.T=0 is checked via trace(U A U.T)=0, while
    U B=0 is checked entrywise.
    """
    count, lam = _count(count), rational(lambda_reg)
    if lam <= 0:
        raise CertificateInputError("lambda_reg must be strictly positive")
    weights = _matrix(weights, "weights")
    d, c = len(weights), len(weights[0]) if weights else 0
    if d < 1 or c < 1:
        raise CertificateInputError("weights must have positive dimensions")
    basis = _matrix(basis, "basis", d)
    rank_dimension = len(basis[0])
    core = _matrix(gram_core, "gram_core", rank_dimension, rank_dimension)
    cross = _matrix(cross_core, "cross_core", rank_dimension, c)
    rank = _prove_psd(core, "gram_core")
    matrices = [("basis", basis), ("gram_core", core), ("cross_core", cross)]
    ub = _matmul(basis, cross, c)
    if not count:
        if any(value for row in ub for value in row):
            raise CertificateInputError("An empty training target must have exactly zero cross moments")
        ua = _matmul(basis, core, rank_dimension)
        trace = sum((ua[i][k] * basis[i][k] for i in range(d)
                     for k in range(rank_dimension)), Fraction(0))
        if trace:
            raise CertificateInputError("An empty training target must have exactly zero Gram moments")
        return _finish("exact-joint-span", matrices, count, lam, weights, weights, rank, empty=True)
    utw = _matmul(_transpose(basis), weights, c)
    small = _matmul(core, utw, c)
    for k in range(rank_dimension):
        for j in range(c):
            small[k][j] -= cross[k][j]
    residual = _matmul(basis, small, c)
    for i in range(d):
        for j in range(c):
            residual[i][j] += lam * count * weights[i][j]
    return _finish("exact-joint-span", matrices, count, lam, weights, residual, rank)
