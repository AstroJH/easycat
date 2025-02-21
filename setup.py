from setuptools import setup, find_packages

setup(
    name="easycat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],
    author="Virjid",
    author_email="astrojh@163.com",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/AstroJH/easycat",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
    python_requires = ">=3.10"
)

