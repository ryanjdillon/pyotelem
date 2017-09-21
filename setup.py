from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyotelem',
    version='0.1',
    description=('Utilities for working with biotelemetry and datalogger '
                 'data in Python'),
    long_description=long_description,
    author='Ryan J. Dillon',
    author_email='ryanjamesdillon@gmail.com',
    url='https://github.com/ryanjdillon/pylleo',
    download_url='https://github.com/ryanjdillon/pyotelem/archive/0.1.tar.gz',
    license='GPL-3.0+',
    packages=['pyotelem'],
    install_requires=[
        'gsw',
        'pandas',
        'scipy',
        'matplotlib'],
    include_package_data=True,
    keywords=['datalogger','accelerometer','biotelemetry'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3.5'],
    zip_safe=False,
    )
