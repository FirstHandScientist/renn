import os
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import zipfile


REPO_DIR = os.path.dirname(os.path.realpath(__file__))


def _findRequirements():
    """Read the requirements.txt file and parse into requirements for setup's
    install_requirements option.
    """
    requirements_path = os.path.join(REPO_DIR, 'requirements.txt')
    try:
        return [line.strip()
                for line in open(requirements_path).readlines()
                if not line.startswith('#')]
    except IOError:
        return []

requirements = _findRequirements()
setup(
    zip_safe=False,
    name='gins',
    version='0.1.0',
    author_email='doli@kth.se',
    description="Inference tool box",
    url='',
    license='',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    classifiers=[  
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'],
    keywords='gins',
)
