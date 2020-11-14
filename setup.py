from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="phoenix_datasets",
    python_requires=">=3.6.0",
    version="0.0.1.dev0",
    description="PyTorch dataset wrappers for PHOENIX 2014 & PHOENIX-2014-T sign language datasets.",
    author="enhuiz",
    author_email="niuzhe.nz@outlook.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["phoenix_datasets"],
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "torchvision",
        "pandarallel",
    ],
    url="https://github.com/enhuiz/phoenix_datasets",
)
