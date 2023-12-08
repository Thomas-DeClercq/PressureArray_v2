#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 17:48:31 2021

@author: pc-robotiq
"""
import os
import numpy

from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
        Extension(
                "PressureReconstruction_onlyPoints",
                ["PressureReconstruction_onlyPoints.pyx"],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()]
            )
        ]



setup(
    ext_modules = cythonize(ext_modules,annotate=True,compiler_directives={'language_level' : "3"})
)

#python3 setup.py build_ext --inplace
