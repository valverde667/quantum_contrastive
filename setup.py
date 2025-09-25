# setup.py
from setuptools import setup, find_packages


def read_requirements(path="requirements.txt"):
    reqs = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "--requirement")):
            # optionally support nested requirement files
            _, nested = line.split(maxsplit=1)
            reqs += read_requirements(nested)
        else:
            reqs.append(line)
    return reqs


setup(
    name="quantum_contrastive",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=read_requirements(),
    include_package_data=True,
)
