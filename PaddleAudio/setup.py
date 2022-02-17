import setuptools

# set the version here
version = '0.1.0a'

with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
    name="PaddleAudio",
    version=version,
    author="",
    author_email="",
    description="PaddleAudio, in development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(exclude=["build*", "test*", "examples*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy >= 1.15.0', 'scipy >= 1.0.0', 'resampy >= 0.2.2',
        'soundfile >= 0.9.0'
    ],
    extras_require={'dev': ['pytest>=3.7', 'librosa>=0.7.2']
                    }  # for dev only, install: pip install -e .[dev]
)
