from setuptools import find_packages, setup

REQUIRED = [
    "tensorflow==2.15.0",
    "hydra-core",
    "optax",
    "tensorflow_datasets",
    "invoke",
    "scipy",
    "wandb",
    "svgutils==0.3.1",
    "invoke",
]

setup(
    name="nex",
    python_requires=">=3.8.0",
    packages=find_packages(),
    install_requires=REQUIRED,
)
