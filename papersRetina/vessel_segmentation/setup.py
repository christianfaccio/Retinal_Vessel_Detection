"""
Setup script for retinal_vessel_segmentation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="retinal-vessel-segmentation",
    version="1.0.0",
    author="Based on Bankhead et al. (2012)",
    description="Retinal vessel detection and measurement using wavelets and edge location refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/retinal-vessel-segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-image>=0.18.0",
        "opencv-python>=4.5.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        "console_scripts": [
            "vessel-segment=demo:main",
        ],
    },
    py_modules=["vessel_segmentation", "visualization", "demo"],
)
