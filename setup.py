# setup.py
from setuptools import setup, find_packages

setup(
    name="quantum_contrastive",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    ],
    include_package_data=True,
)
