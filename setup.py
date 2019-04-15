#!/usr/bin/env python
from distutils.core import setup
import numpy
from Cython.Build import cythonize

setup(
  name = 'calc_score',
  ext_modules = cythonize("./rethinknet/calc_score.pyx",
                          compiler_directives={'language_level': 3}),
  include_dirs=[numpy.get_include()]
)
