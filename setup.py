from setuptools import setup
import os

VERSION = "0.1"


def get_long_description():
    with open(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"),
        encoding="utf8",
    ) as fp:
        return fp.read()


setup(
    name="talk-summarizer",
    description="Python library to summarize talks",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="Lawrence Wu",
    url="https://github.com/lawwu/talk-summarizer",
    project_urls={
        "Issues": "https://github.com/lawwu/talk-summarizer/issues",
        "CI": "https://github.com/lawwu/talk-summarizer/actions",
        "Changelog": "https://github.com/lawwu/talk-summarizer/releases",
    },
    license="Apache License, Version 2.0",
    version=VERSION,
    packages=["talk_summarizer"],
    install_requires=[],
    extras_require={"test": ["pytest"]},
    python_requires=">=3.7",
)
