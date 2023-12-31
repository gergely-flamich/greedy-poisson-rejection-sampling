import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="greedy-poisson-rejection-sampling",
    version="0.0.1",
    author="Gergely Flamich",
    author_email="gf332@cam.ac.uk",
    description="Greedy Poisson Rejection Sampling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gergely-flamich/greedy-poisson-rejection-sampling",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "scipy"
    ]
)