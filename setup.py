from setuptools import setup, find_packages


setup(
    name="motion_blur",
    use_scm_version=True,
    install_requires=["numpy", "matplotlib", "pyyaml", "torch", "torchvision"],
    extras_require={"TEST_SUITE": ["pytest", "black==19.10b0", "pylint", "flake8"]},
    packages=find_packages(),
)
