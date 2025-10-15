#!/usr/bin/env python3
"""
Setup script for F1 Infringement Analysis
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="f1-infringement-analysis",
    version="1.0.0",
    author="MSIM Text Mining Team",
    author_email="pranavh@umich.edu",
    description="F1 Infringement Analysis: Multi-Team NLP Summarization Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/f1-infringement-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "f1-analyze=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "config/*.json"],
    },
)
