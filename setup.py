from setuptools import setup, find_packages


setup(
    name="motion_blur",
    packages=find_packages(),
    use_scm_version=True,
    install_requires=["numpy", "matplotlib"],
    extras_require={"TEST_SUITE": ["pytest","black==19.10b0", "pylint", "flake8"]},
)
