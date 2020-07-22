from setuptools import setup, find_packages

setup(
    name="mbr_nmt",
    version="0.1dev",
    description="minimum Bayes-risk decoding for neural machine translation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Bryan Eikema",
    packages=find_packages(),
    entry_points={
        'console_scripts': ['mbr-nmt=mbr_nmt.mbr_nmt:main'],
    }
)

