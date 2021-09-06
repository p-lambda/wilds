import setuptools
import os
import sys

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, 'wilds'))
from version import __version__

print(f'Version {__version__}')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wilds",
    version=__version__,
    author="WILDS team",
    author_email="wilds@cs.stanford.edu",
    url="https://wilds.stanford.edu",
    description="WILDS distribution shift benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires = [
        'numpy>=1.19.1',
        'ogb>=1.2.6',
        'outdated>=0.2.0',
        'pandas>=1.1.0',
        'pillow>=7.2.0',
        'pytz>=2020.4',
        'torch>=1.7.0',
        'torchvision>=0.8.2',
        'tqdm>=4.53.0',
        'scikit-learn>=0.20.0',
        'scipy>=1.5.4'
    ],
    license='MIT',
    packages=setuptools.find_packages(exclude=['dataset_preprocessing', 'examples', 'examples.models', 'examples.models.bert']),
    classifiers=[
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
)
