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
              'seismic demultiple'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Ulises Berman, Nick Luiken, Matteo Ravasi',
    author_email='ulises.berman@kaust.edu.sa, n=icholas.luiken@kaust.edu.sa, matteo.ravasi@kaust.edu.sa',
    install_requires=['numpy>=1.18.4',
                      'torch>=2.0.1',
                      'pylops==1.18.3',
                      'devito>=4.7.1',
                      'kymatio>=0.3.0',
                      'tqdm>=4.65.0',
                      'cupy>=8.3.0'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('adasubtraction/version.py')),
    setup_requires=['setuptools_scm'],

)
