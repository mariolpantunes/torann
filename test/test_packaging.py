"""Tests that `requirements.txt` and `pyproject.toml` agree.

They have drifted before -- the numpy floor read 2.4.0 in one file and 2.0.0
in the other -- and nothing breaks when they do, so nothing surfaces it. This
is the assertion that does.
"""

import os
import tomllib
import unittest

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)


def _name_of(spec):
    """`"numpy>=2.0.0"` -> `"numpy"`."""
    for sep in ("==", ">=", "<=", "~=", ">", "<", "["):
        spec = spec.split(sep)[0]
    return spec.strip().lower()


def _requirements():
    """`name -> full specifier`, skipping comments and blanks."""
    out = {}
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split("#", 1)[0].strip()
            if line:
                out[_name_of(line)] = line
    return out


class TestPackaging(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
            self.project = tomllib.load(handle)["project"]
        self.declared = _requirements()

    def test_requirements_covers_every_runtime_dependency(self):
        missing = {_name_of(d) for d in self.project["dependencies"]} - set(self.declared)
        self.assertFalse(
            missing, f"in pyproject.toml but not requirements.txt: {sorted(missing)}")

    def test_runtime_floors_agree_between_the_two_files(self):
        for spec in self.project["dependencies"]:
            with self.subTest(dependency=spec):
                self.assertEqual(spec, self.declared[_name_of(spec)])
