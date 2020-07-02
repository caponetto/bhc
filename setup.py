# -*- coding: utf-8 -*-

# License: GPL 3.0

import setuptools

import versioneer

NAME = "bhc"
AUTHOR = "Guilherme Caponetto"
URL = "https://github.com/caponetto/bhc"
LICENSE = "GPL-3.0"
DESCRIPTION = "Python implementation of Bayesian hierarchical clustering (BHC) and Bayesian rose trees (BRT) algorithms."

LONG_DESCRIPTION = """
This is a python implementation of the following algorithms.

Bayesian Hierarchical Clustering algorithm proposed by Heller & Ghahramani (2005).
> HELLER, Katherine A.; GHAHRAMANI, Zoubin. Bayesian hierarchical clustering. In: **Proceedings of the 22nd international conference on Machine learning**. 2005. p. 297-304.

Bayesian Rose Trees proposed by Blundell et al (2012).
> BLUNDELL, Charles; TEH, Yee Whye; HELLER, Katherine A. Bayesian rose trees. arXiv preprint arXiv:1203.3468, 2012.
"""

REQUIREMENTS = ["numpy==1.18.5", "scipy==1.5.0"]

CLASSIFIERS = [
    "Environment :: Console",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
]

setuptools.setup(name=NAME,
                 version=versioneer.get_version(),
                 cmdclass=versioneer.get_cmdclass(),
                 description=DESCRIPTION,
                 long_description=LONG_DESCRIPTION,
                 long_description_content_type="text/markdown",
                 url=URL,
                 author=AUTHOR,
                 author_email="N/A",
                 include_package_data=True,
                 license=LICENSE,
                 install_requires=REQUIREMENTS,
                 packages=setuptools.find_packages(),
                 platforms="any",
                 classifiers=CLASSIFIERS,
                 python_requires='>=3.6')
