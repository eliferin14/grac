from setuptools import setup, find_packages

setup(
    name="gesture_control",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    zip_safe=True,
)