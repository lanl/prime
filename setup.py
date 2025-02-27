""" PNLP """
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Read the contents of requirements.txt
with open(path.join(here, 'requirements', 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

PYTHON_REQUIRES = '>=3.7.*'    
setup(
    name='pnlp',
    version='0.1',
    description='Protein sequence NLP',
    url='',
    author='Bin Hu, Michal Babinski, Kaetlyn Gibson',
    author_email='{bhu, mbabinski, kaetlyn}@lanl.gov',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: BSD',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    keywords='Protein, natural language processing, NLP',
    packages=find_packages('src', exclude=['contrib', 'docs', 'tests']),
    package_dir={'': 'src'},
    install_requires=requirements,
)
