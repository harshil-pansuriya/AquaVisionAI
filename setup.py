# AquaVisionAI/setup.py
from setuptools import setup, find_packages

# Read requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="AquaVisionAI",
    version="0.1.0",
    description="Automated Analysis of Underwater Imagery for Marine Ecosystem Monitoring using AI",
    author="Harshil",
    packages=find_packages(exclude=["tests", "notebooks"]),
    install_requires=requirements,
    python_requires=">=3.9,<=3.12",
    include_package_data=True,
    package_data={
        "config": ["*.yaml"],
    },
)