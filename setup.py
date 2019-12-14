from setuptools import setup, find_packages
import os


def requirements(fp: str):
    with open(fp) as f:
        return [r.strip() for r in f.readlines()]


fp_readme = os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.rst")
with open(fp_readme, encoding="utf-8") as f:
    long_description = f.read()

setup_requirements = ["pytest-runner", "setuptools_scm"]

setup(
    name="pyotelem",
    description="A library for processing biotelemetry and datalogger data",
    author="Ryan J. Dillon",
    author_email="ryanjamesdillon@gmail.com",
    url="https://github.com/ryanjdillon/pyotelem",
    license="GPL-3.0+",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=setup_requirements,
    install_requires=requirements("requirements.in"),
    tests_require=requirements("requirements_test.txt"),
    include_package_data=True,
    use_scm_version={"write_to": "src/pyotelem/_version.py", "relative_to": __file__},
    keywords=["datalogger", "accelerometer", "biotelemetry"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=True,
)
