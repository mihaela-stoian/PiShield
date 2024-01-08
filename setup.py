import os
from setuptools import find_packages, setup

# Package meta-data.
NAME = 'CLOVERD'
REQUIRES_PYTHON = '>=3.8.0'
VERSION = '0.0.1'
DESCRIPTION = 'Building Constraint Layers over Deep Neural Networks'

# required packages
REQUIRED = [
    'numpy',
    'torch'
]

path_dir = os.path.abspath(os.path.dirname(__file__))
long_description = DESCRIPTION

descr = {}
if not VERSION:
    project_ver = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(path_dir, project_ver, '__version__.py')) as f:
        exec(f.read(), descr)
else:
    descr['__version__'] = VERSION

setup(
    name=NAME,
    version=descr['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*",
                                    "examples", "*.examples", "*.examples.*", "examples.*",
                                    "data", "*.data", "*.data.*", "data.*",
                                    "build", "*.build", "*.build.*", "build.*"]),
    project_urls={
        'CLOVERD github repo': ''
    },
    install_requires=REQUIRED,
    include_package_data=True,
)
