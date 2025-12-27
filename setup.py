from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recallbricks-langchain",
    version="1.3.0",
    author="RecallBricks",
    author_email="tyler.kutscher@gmail.com",
    description="Enterprise-grade persistent memory for LangChain agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/recallbricks/recallbricks-langchain",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.31.0",
        "langchain>=0.1.0",
    ],
)
