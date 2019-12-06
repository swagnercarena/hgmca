#!/usr/bin/env python

try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

from setuptools import find_packages
import os

readme = open("README.rst").read()

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

required_packages = ['numpy>=1.16.2','numba>=0.43.1']
tests_required_packages = ['pytest>=2.3']

setup(
    name='hgmca',
    version='1.0.0',
    description='Hiearchical component seperation package using sparsity.',
    long_description=readme,
    long_description_content_type='text/markdown',
    license='LICENSE.md',
    author='Sebastian Wagner-Carena',
    author_email='sebaswagner@outlook.com',
    url='https://github.com/swagnercarena/hgmca',
    packages=find_packages(PACKAGE_PATH),
    package_dir={'hgmca': 'hgmca'},
    include_package_data=True,
    install_requires=required_packages,
    zip_safe=False,
    test_suite='nose.collector',
    tests_require=tests_required_packages
)