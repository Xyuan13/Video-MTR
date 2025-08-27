from setuptools import setup, find_packages
import os

setup(
    name="video-mtr",
    version="0.1.0",
    packages=find_packages(),
    python_requires='>=3.10',
    author="Xyuan13",
    description="A Python package for Video-MTR",
    long_description=open("README.md", "r").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
