#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:48:31 2021

@author: pc-robotiq
"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("ValidateFromFEM_roundedSurface.pyx",annotate=True,compiler_directives={'language_level' : "3"})
)

#python3 setup.py build_ext --inplace