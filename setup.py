from setuptools import setup, find_packages

setup(
    name="chain_processor",
    version="0.1.0",
    description="A library for creating and managing chains of function nodes.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Francesco Lorè",
    author_email="flore9819@gmail.com",
    url="https://github.com/lf-data/chain_processor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
