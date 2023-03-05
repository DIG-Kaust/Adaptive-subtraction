import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Adaptive subtraction with an L1-L1 optimization solver.'

setup(
    name="adasubtraction",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'optimization',
              'seismic'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Nick Luiken, Ulises Berman, Matteo Ravasi',
    author_email='nicholas.luiken@kaust.edu.sa, ulises.berman@kaust.edu.sa, matteo.ravasi@kaust.edu.sa',
    install_requires=['numpy >= 1.15.0',
                      'torch >= 1.2.0',
                      'pylops == 1.18.3'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('adasubtraction/version.py')),
    setup_requires=['setuptools_scm'],

)