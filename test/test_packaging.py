"""Tests that `requirements.txt` and `pyproject.toml` agree.

They have drifted before -- the numpy floor read 2.4.0 in one file and 2.0.0
in the other -- and nothing breaks when they do, so nothing surfaces it. This
is the assertion that does.

The same holds for the version and contact the package reports: pyBlindOpt
shipped 0.3.0 saying it was 0.2.0, from a hand-kept copy nothing checked.
"""

import importlib.metadata
import os
import re
import tomllib
import unittest

import torann

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _name_of(spec):
    """`"numpy>=2.0.0"` -> `"numpy"`."""
    return re.split(r"[<>=~!\[ ]", spec, maxsplit=1)[0].strip().lower()


def _requirements():
    """`name -> full specifier`."""
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        return {_name_of(line): line.strip() for line in handle if line.strip()}


class TestPackaging(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            self.dependencies = tomllib.load(handle)["project"]["dependencies"]
        self.declared = _requirements()

    def test_requirements_covers_every_runtime_dependency(self):
        missing = {_name_of(d) for d in self.dependencies} - set(self.declared)
        self.assertFalse(
            missing, f"in pyproject.toml but not requirements.txt: {sorted(missing)}")

    def test_requirements_declares_nothing_extra(self):
        """It is the runtime dependency list, not a development environment."""
        extra = set(self.declared) - {_name_of(d) for d in self.dependencies}
        self.assertFalse(
            extra, f"in requirements.txt but not pyproject.toml: {sorted(extra)}")

    def test_runtime_floors_agree_between_the_two_files(self):
        for spec in self.dependencies:
            with self.subTest(dependency=spec):
                self.assertEqual(spec, self.declared[_name_of(spec)])


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
