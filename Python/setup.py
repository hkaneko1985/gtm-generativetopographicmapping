# -*- coding: utf-8 -*-
from setuptools import setup, find_packages


with open('../README.md') as f:
    readme = f.read()

with open('../LICENSE') as f:
    license = f.read()

setup(
    name='gtm',
    version='1.0.0',
    description='Generative Topographic Mapping',
    long_description=readme,
    author='Hiromasa Kaneko',
    author_email='hkaneko226@gmail.com',
    url='https://github.com/hkaneko1985/gtm-generativetopographicmapping',
    license=license,
    install_requires=['numpy', 'scikit-learn', 'scipy'],
    packages=find_packages()
)

