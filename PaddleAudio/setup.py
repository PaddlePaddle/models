
import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PaddleAudio", 
    version="0.0.0",
    author="",
    author_email="",
    description="PaddleAudio, in development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ranchlai/PaddleAudio",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
