"""Versioned exact logistic verification with dyadic integer accumulators.

The original rational sigmoid enclosure and work caps remain unchanged.
The result has exactly the original coordinate gradient endpoints and bounds.
Only accumulation changes. This is not an encoder or approximation backend.
"""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

from phase4 import convex as original
from phase4.requests import digest
from phase5.convex_multioutput import _inputs as multi_inputs

BACKEND = 'exact_dyadic_integer_v1'


def dyadic_vector(values):
    """Exact common binary denominator. Only one input row need be resident."""
    ratios = [float(v).as_integer_ratio() for v in values]
    exponent = max((denominator.bit_length() - 1 for _, denominator in ratios), default=0)
    return [numerator << (exponent - denominator.bit_length() + 1) for numerator, denominator in ratios], exponent


def integer_bytes(values):
    """Shallow byte sum by reference; shared integer objects can count repeatedly."""
    return sys.getsizeof(values) + sum(sys.getsizeof(value) for value in values)


def certify_logistic(features, targets, lambda_reg, weights, *, bits=128, max_terms=256,
                     max_squarings=64, max_coordinates=2_000_000,
                     parameter_tolerance=Fraction(1, 10**8)):
    started = time.perf_counter()
    x, y, w = original._inputs(features, targets, weights)
    lam_float = original._positive(lambda_reg, 'lambda_reg')
    lam = original.rational(lam_float)
    cap = original._integer(max_coordinates, 'max_coordinates', 0)
    n, d = x.shape
    bits = original._integer(bits, 'bits')
    max_terms = original._integer(max_terms, 'max_terms')
    max_squarings = original._integer(max_squarings, 'max_squarings', 0)
    if bits > 4096:
        raise original.CertificateBudgetError('Requested dyadic precision exceeds hard verification cap')
    if n * d > cap:
        raise original.CertificateBudgetError('Exact gradient row-coordinate budget exceeded before evaluation')
    tolerance = original.rational(parameter_tolerance)
    if tolerance < 0:
        raise ValueError('Parameter tolerance must be nonnegative')
    wi, weight_exponent = dyadic_vector(w)
    lam_numerator, lam_denominator = lam_float.as_integer_ratio()
    exponent = weight_exponent + lam_denominator.bit_length() - 1
    count_denominator = max(1, n)
    lower = [lam_numerator * value * count_denominator for value in wi]
    upper = lower.copy()
    counts = {'exact_logit_row_evaluations': 0, 'exact_feature_coordinates': 0,
              'taylor_terms': 0, 'squarings': 0}
    rescalings = 0
    tracked_bytes = integer_bytes(wi) + integer_bytes(lower) + integer_bytes(upper)
    max_logit_bits = 0
    for row, target in zip(x, y):
        xi, feature_exponent = dyadic_vector(row)
        numerator = sum(a * b for a, b in zip(xi, wi))
        max_logit_bits = max(max_logit_bits, abs(numerator).bit_length())
        logit = Fraction(numerator, 1 << (feature_exponent + weight_exponent))
        a, b, cost = original.sigmoid_interval(logit, bits=bits, max_terms=max_terms,
                                              max_squarings=max_squarings)
        counts['exact_logit_row_evaluations'] += 1
        counts['exact_feature_coordinates'] += d
        counts['taylor_terms'] += cost['taylor_terms']
        counts['squarings'] += cost['squarings']
        target_numerator, target_denominator = float(target).as_integer_ratio()
        target_exponent = target_denominator.bit_length() - 1
        residual_exponent = max(bits, target_exponent)
        low_residual = (a.numerator << (residual_exponent - a.denominator.bit_length() + 1)) - (target_numerator << (residual_exponent - target_exponent))
        high_residual = (b.numerator << (residual_exponent - b.denominator.bit_length() + 1)) - (target_numerator << (residual_exponent - target_exponent))
        row_exponent = feature_exponent + residual_exponent
        if row_exponent > exponent:
            shift = row_exponent - exponent
            lower = [value << shift for value in lower]
            upper = [value << shift for value in upper]
            exponent = row_exponent
            rescalings += 1
        shift = exponent - row_exponent
        for j, value in enumerate(xi):
            left, right = (low_residual, high_residual) if value >= 0 else (high_residual, low_residual)
            lower[j] += (value * left) << shift
            upper[j] += (value * right) << shift
        tracked_bytes = max(tracked_bytes, integer_bytes(wi) + integer_bytes(xi) + integer_bytes(lower) + integer_bytes(upper))
    denominator = count_denominator << exponent
    lower_rational = [Fraction(value, denominator) for value in lower]
    upper_rational = [Fraction(value, denominator) for value in upper]
    # The common denominator permits one exact squared-norm sum.
    squared_numerator = sum(max(abs(a), abs(b)) ** 2 for a, b in zip(lower, upper))
    gradient_squared = Fraction(squared_numerator, denominator * denominator)
    parameter_squared = gradient_squared / (lam * lam)
    radius = original.sqrt_upper_rational(parameter_squared, 30)
    gap = gradient_squared / (2 * lam)
    result = {'schema': 'ccu-fractional-logistic-dyadic-certificate-v1',
              'verification_backend': BACKEND,
              'status': 'certified_stored_value_optimization_bound',
              'objective': 'single-output mean fractional BCE + lambda/2 * squared Euclidean weight norm; no intercept',
              'empty_target': 'zero data loss; unique minimizer zero',
              'records': n, 'dimension': d, 'lambda_exact': original._fraction_json(lam),
              'sigmoid_interval_precision_bits': bits,
              'gradient_lower': [original._fraction_json(v) for v in lower_rational],
              'gradient_upper': [original._fraction_json(v) for v in upper_rational],
              'gradient_norm_squared_upper': original._fraction_json(gradient_squared),
              'parameter_error_squared_upper': original._fraction_json(parameter_squared),
              'parameter_error_norm_upper': original._fraction_json(radius),
              'objective_gap_upper': original._fraction_json(gap),
              'parameter_tolerance': original._fraction_json(tolerance),
              'meets_parameter_tolerance': parameter_squared <= tolerance * tolerance,
              'radius_float_for_display_only': float(radius), 'work': counts,
              'dyadic_work': {'accumulator_rescalings': rescalings,
                              'final_common_binary_exponent': exponent,
                              'max_logit_integer_bits': max_logit_bits,
                              'max_gradient_integer_bits': max((abs(v).bit_length() for v in lower + upper), default=0),
                              'tracked_integer_list_shallow_byte_sum': tracked_bytes,
                              'memory_scope': 'sum of list storage and integer size per reference; shared integers can count repeatedly; excludes conversion temporaries, sigmoid workspace, inputs, and output serialization; neither a lower bound nor an upper bound on total memory',
                              'memory_metadata_revision': 2,
                              'integer_products_for_logits': n * d,
                              'integer_products_for_gradient_endpoints': 2 * n * d,
                              'rational_coordinate_accumulation_operations': 0},
              'elapsed_seconds': time.perf_counter() - started,
              'input_bindings': {'features_sha256': original._hash_array(x),
                                 'targets_sha256': original._hash_array(y),
                                 'weights_sha256': original._hash_array(w),
                                 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'original_sigmoid_code_sha256': hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest()},
              'statistical_unlearning_certificate': False, 'privacy_claim': False,
              'canonical_state_claim': False, 'peak_memory_measured': False,
              'resource_limits': {'max_coordinates': cap, 'max_terms': max_terms,
                                  'max_squarings': max_squarings}}
    result['sha256'] = digest(result)
    return result


def certify_multioutput(features, targets, lambda_reg, weights, *, max_coordinates=2_000_000,
                        parameter_tolerance=Fraction(1, 10**8), **certificate_options):
    x, y, w = multi_inputs(features, targets, weights)
    started = time.perf_counter()
    cap = original._integer(max_coordinates, 'max_coordinates', 0)
    if x.size * y.shape[1] > cap:
        raise original.CertificateBudgetError('Total output-row-coordinate budget exceeded before verification')
    tolerance = original.rational(parameter_tolerance)
    if tolerance < 0:
        raise ValueError('nonnegative Frobenius tolerance required')
    certificates = [certify_logistic(x, y[:, c], lambda_reg, w[:, c], max_coordinates=cap,
                                   parameter_tolerance=tolerance, **certificate_options)
                    for c in range(y.shape[1])]
    to_fraction = lambda value: Fraction(int(value['numerator']), int(value['denominator']))
    squared = sum((to_fraction(c['parameter_error_squared_upper']) for c in certificates), Fraction(0))
    gap = sum((to_fraction(c['objective_gap_upper']) for c in certificates), Fraction(0))
    radius = original.sqrt_upper_rational(squared, 30)
    work = {key: sum(c['work'][key] for c in certificates) for key in certificates[0]['work']}
    result = {'schema': 'ccu-multioutput-logistic-dyadic-certificate-v1',
              'verification_backend': BACKEND, 'outputs': y.shape[1], 'records': len(x),
              'objective_normalization': 'sum_over_outputs_mean_over_rows',
              'scalar_certificates': certificates,
              'parameter_error_frobenius_squared_upper': original._fraction_json(squared),
              'parameter_error_frobenius_upper': original._fraction_json(radius),
              'objective_gap_upper': original._fraction_json(gap),
              'parameter_tolerance': original._fraction_json(tolerance),
              'meets_parameter_tolerance': squared <= tolerance * tolerance,
              'radius_float_for_display_only': float(radius), 'work': work,
              'total_coordinate_budget': cap,
              'input_bindings': {name: original._hash_array(a) for name, a in [('features', x), ('targets', y), ('weights', w)]},
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'original_sigmoid_code_sha256': hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
              'elapsed_seconds': time.perf_counter() - started,
              'statistical_unlearning_certificate': False, 'canonical_state_claim': False, 'privacy_claim': False}
    result['sha256'] = digest(result)
    return result
