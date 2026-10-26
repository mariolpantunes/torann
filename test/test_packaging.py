"""Tests that the package reports what `pyproject.toml` publishes.

pyBlindOpt shipped 0.3.0 saying it was 0.2.0, from a hand-kept copy nothing
checked. `pyproject.toml` is the only manifest; there is no
`requirements.txt` to drift from it.
"""

import importlib.metadata
import os
import tomllib
import unittest

import torann

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _installed():
    """Is this distribution installed, or are we running from a bare checkout?"""
    try:
        importlib.metadata.version("torann")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


class TestReportedMetadata(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            self.project = tomllib.load(handle)["project"]

    @unittest.skipUnless(_installed(), "no distribution metadata to compare against")
    def test_reported_version_matches_the_published_one(self):
        """What the package reports is what pyproject.toml declares.

        Only meaningful against an installed distribution -- the version is
        read from its metadata. A bare checkout (CI's pure-Python job runs
        one) reports the dev fallback instead, which is asserted below.
        """
        self.assertEqual(torann.__version__, self.project["version"])

    @unittest.skipIf(_installed(), "distribution is installed")
    def test_a_bare_checkout_reports_the_dev_fallback(self):
        self.assertEqual(torann.__version__, "0.0.0.dev0")

    def test_contact_address_is_the_one_pyproject_publishes(self):
        self.assertEqual(torann.__email__, self.project["authors"][0]["email"])


_REQUIRE_NATIVE = os.environ.get("TORANN_REQUIRE_NATIVE")


class TestBuildUnderTest(unittest.TestCase):
    """The run tests the build it claims to.

    A missing `_native` module is not an error -- the package falls back to the
    pure-Python backend and every native test skips. So a gate that forgot to
    build the extension in place passed green while testing none of it, and
    CI's native job did exactly that: it installed the wheel into site-packages
    and ran the suite against the source tree, which has no `.so`. The gate
    sets `TORANN_REQUIRE_NATIVE` (1 for the native run, 0 for the pure-Python
    one), and this fails when the run is not what it says.
    """

    @unittest.skipUnless(_REQUIRE_NATIVE, "TORANN_REQUIRE_NATIVE is not set")
    def test_native_extension_is_what_the_run_requires(self):
        from torann import rust

        self.assertEqual(rust.AVAILABLE, _REQUIRE_NATIVE == "1", f"torann.rust.AVAILABLE is {rust.AVAILABLE}")
