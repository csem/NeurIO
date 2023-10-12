#!/user/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_packages = f.read().splitlines()

setup(name='neurio',
      version='0.0.1',
      author='Simon Narduzzi',
      description='A benchmarking library for neuromorphic hardware platforms.',
      long_description='A benchmarking library for neuromorphic hardware platforms.',
      url='https://github.com/Narduzzi',
      keywords='development, setup, setuptools',
      python_requires='>=3.7, <4',
      packages=find_packages("neurio"),
      package_dir={"": "neurio"},
      install_requires=required_packages,
      entry_points={
          'runners': [
              'sample=sample:main',
          ]
      }
      )
