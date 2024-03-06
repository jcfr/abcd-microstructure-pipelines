from __future__ import annotations

import importlib.metadata

import abcd_microstructure_pipelines as m


def test_version():
    assert importlib.metadata.version("abcd_microstructure_pipelines") == m.__version__
