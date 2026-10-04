Exact chart helper (local execution backend)

Compile in this workspace:
  g++ -std=c++17 -O2 exact_chart.cpp -Wl,-l:libgmp.so.10 -o exact_chart
  ./exact_chart --self-test

The current container has g++ and GMP 6.3.0's shared runtime but no GMP or Boost
headers. The helper declares the documented GMP integer/rational ABI explicitly
for LP64 Linux, with compile-time sizeof checks. This is a platform-specific
acceleration backend, not a new arbitrary-precision implementation. It uses
GMP-owned rational initialization, canonical arithmetic, string conversion and
cleanup. It does not change the installed library or use application data files.
Input and output are exact textual rational factors over stdin/stdout.

Public references consulted 2026-10-04:
  https://gmplib.org/manual/Integer-Internals
  https://gmplib.org/manual/Rational-Internals
  https://gmplib.org/manual/Initializing-Rationals
  https://gmplib.org/manual/Rational-Arithmetic

GMP documents its internals as potentially version-dependent; do not treat this
headerless ABI binding as portable to other architectures or future runtimes.
The startup test checks canonical rational addition, multiplication and division.
The current build reported:
  GMP 6.3.0 exact rational ABI and arithmetic checks passed

ccu/exact_canonical.py has an exact pure-Python fractions.Fraction fallback.
Pass backend='python' to build/delete/from_core to select it. Native and Python
results matched the 20 algebra-only fixtures in check_exact_algebra.py, including
rank deficiency and complete Gram cancellation with a surviving cross moment.
Canonical checkpoint bytes contain no backend identifier; the backend cannot
change the represented state. Build and merge have bounded rank cores, but
arbitrary-precision integer bit growth and temporary arithmetic remain real costs.
