import os
import setuptools

def read(fname: str) -> str:
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

long_description = read("README.md")
install_requires = read("requirements.txt")

setuptools.setup(
    name="ick",
    version="0.0.1",
    author="Ziyang Jiang",
    author_email="ziyang.jiang@duke.edu", 
    description="A Python implementation of Implicit Composite Kernel (Jiang et al. 2022)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jzy95310/ICK",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
)