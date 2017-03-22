#!/usr/bin/env python
"""conftest.py: General configuration file for tests"""
__author__ = "Frederic Bergeron"
__license__ = "undecided"
__version__ = "1.0"
__email__ = "fbergeron@pa.jst.jp"
__status__ = "Development"

import pytest


def pytest_addoption(parser):
    parser.addoption("--gpu", default=None, action="store",
                     help="Specify which gpu to use to perform the tests. Otherwise, use the cpu if the parameter is omitted.")


@pytest.fixture
def gpu(request):
    return request.config.getoption('--gpu')
