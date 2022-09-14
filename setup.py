#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="min-opt",
    version="0.0.1",
    description="MinOPT model repository",
    author="Benjamin Bolte",
    url="https://github.com/treasure-robotics/minopt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "torch>=1.12",
        "transformers",
        "tqdm",
    ],
    extras_require={
        "dev": {
            "black",
            "darglint",
            "flake8",
            "mypy-extensions",
            "mypy",
            "pylint",
            "pytest",
            "types-setuptools",
            "typing_extensions",
        },
    },
    python_requires=">=3.10",
)
