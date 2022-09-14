#!/usr/bin/env python

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="minopt",
    version="0.0.1",
    description="MinOPT model repository",
    author="Benjamin Bolte",
    url="https://github.com/Treasure-Robotics/minOPT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        "torch>=1.12",
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
            "transformers",
            "types-setuptools",
            "typing_extensions",
        },
        "scripts": {
            "transformers",
        },
    },
    package_data={
        "minopt": ["py.typed"],
    },
    python_requires=">=3.10",
)
