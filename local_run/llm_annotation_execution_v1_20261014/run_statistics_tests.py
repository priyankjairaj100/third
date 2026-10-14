#!/usr/bin/env python3
"""Relocate only the sealed statistics test's source-root variable for private staging."""
import argparse
from pathlib import Path
import sys
import unittest
import run_model as R


def suite(folder,repo,expected_sha):
    """No production statistics functions or test assertions are changed."""
    folder=Path(folder);pin=R.binding(folder/'package_manifest.json')
    R.require(pin['sha256']==expected_sha,'statistics_test_package_pin');R.package(folder,pin)
    R.exact_import('statistics_core',folder/'statistics_core.py')
    tests=R.exact_import('third_relocated_statistics_tests',folder/'test_statistics.py')
    tests.REPO=Path(repo)
    return unittest.defaultTestLoader.loadTestsFromModule(tests)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--statistics-package',type=Path,required=True)
    p.add_argument('--statistics-package-sha256',required=True)
    p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args();result=unittest.TextTestRunner(verbosity=2).run(suite(a.statistics_package,a.repo,a.statistics_package_sha256))
    return 0 if result.wasSuccessful() and result.testsRun>0 and not result.skipped else 2


if __name__=='__main__':raise SystemExit(main())
