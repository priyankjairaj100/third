"""Canonical bytes in/out, with an exact residual gate on each released head.

The state is the canonical byte string only. Candidate heads, certificates,
solver workspaces and reconstructed indexes are not persisted in that state.
No original feature/label table is accepted or read by repair/release.
"""
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import numpy as np
from .exact_canonical import ExactCanonicalSummary
from .certified_ridge import factor_residual_certificate, rational, RidgeCertificate


class CertificationFailure(RuntimeError):
    """No model may be released at the requested tolerance."""


@dataclass(frozen=True)
class StrictRelease:
    canonical_state: bytes
    state_sha256: str
    weights: np.ndarray
    certificate: RidgeCertificate


def repair_and_release(state_bytes: bytes, identifiers, lambda_reg=0.01,
                       tolerance=Fraction(1,10**10), *, validate_state=True,
                       backend='auto') -> StrictRelease:
    """Return a certified release or raise, leaving input state bytes untouched.

    The input is a strict exact checkpoint, never a legacy floating checkpoint.
    Infeasible tolerances are an explicit failure, not a silent relaxation.
    The supplied lambda retains its exact binary/rational meaning in the check.
    """
    tol=rational(tolerance)
    if tol<0:raise ValueError('tolerance must be nonnegative')
    state=ExactCanonicalSummary.from_bytes(state_bytes,validate=validate_state)
    state.delete(identifiers,backend=backend)
    weights=state.decode_candidate(lambda_reg)
    if not np.isfinite(weights).all():
        raise CertificationFailure('candidate is nonfinite; no release')
    certificate=factor_residual_certificate(*state.factor(),lambda_reg,weights)
    if not certificate.meets_tolerance(tol):
        raise CertificationFailure('exact residual bound exceeds requested tolerance; no release')
    encoded=state.canonical_bytes()
    return StrictRelease(encoded,hashlib.sha256(encoded).hexdigest(),weights,certificate)
