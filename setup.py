#!/usr/bin/env python
#
# Copyright 2016, Noel Burton-Krahn <noel@burton-krahn.com>
# All Rights Reserved.
#

import os
import setuptools

INSTALL_ROOT = os.environ.get("VIRTUAL_ENV", "")

setuptools.setup(
    name='genetic',
    version='0.1',
    description='Genetic Programming',
    author='Noel Burton-Krahn',
    packages=[],
    install_requires=[],
    tests_require=['nose'],
    test_suite='nose.collector',
    py_modules=['genetic'],
    )
