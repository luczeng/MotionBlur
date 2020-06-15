from setuptools import setup, find_packages


setup(
    name="motion_blur",
    use_scm_version=True,
    install_requires=[
        "torchsummary",
        "scikit-image",
        "opencv-python",
        "numpy",
        "matplotlib",
        "pyyaml",
        "torch",
        "torchvision",
        "wget",
    ],
    extras_require={
        "TEST_SUITE": ["pytest", "black==19.10b0", "pylint", "flake8"],
        "DEVELOP": ["Tensorboard", "mlflow"],
    },
    packages=find_packages(),
)
