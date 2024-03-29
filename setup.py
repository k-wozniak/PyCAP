import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyCAP",
    version="0.0.1",
    author="Kamil Wozniak",
    author_email="kkw28@bath.ac.uk.com",
    description="Tools based on 2CAP model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    project_urls={
        "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages = setuptools.find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'numba',
        'osqp',
    ],
)