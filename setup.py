#!/usr/bin/env python3
"""
Setup script for Kibali Language Framework
"""

from setuptools import setup, find_packages
import os

# Read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="kibali-langage",
    version="1.0.0",
    author="Lojol469",
    author_email="lojol469@gmail.com",
    description="Écosystème de Nano-IA Vivantes - Langage organique pour IA autonomes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lojol469-cmd/kibalone-langage",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Interpreters",
    ],
    keywords="ai artificial-intelligence llm rag autonomous-cells organic-programming",
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.2.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "sphinx>=5.0.0",
        ],
        "gpu": [
            "torch[cu118]>=2.0.0",  # CUDA 11.8
        ],
    },
    entry_points={
        "console_scripts": [
            "kibali=kibali_cmd:main",
        ],
    },
    include_package_data=True,
    package_data={
        "kibali": [
            "cells/*.kib",
            "rag/config.json",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/lojol469-cmd/kibalone-langage/issues",
        "Source": "https://github.com/lojol469-cmd/kibalone-langage",
        "Documentation": "https://github.com/lojol469-cmd/kibalone-langage/blob/main/README_RAG_3D.md",
    },
)