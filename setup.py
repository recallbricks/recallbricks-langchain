from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recallbricks-langchain",
    version="0.1.0",
    description="LangChain integration for RecallBricks Memory Graph",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RecallBricks",
    author_email="support@recallbricks.com",
    url="https://github.com/recallbricks/recallbricks-langchain",
    packages=find_packages(),
    install_requires=[
        "requests>=2.25.0",
        "langchain>=0.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "examples": [
            "langchain-openai>=0.0.1",
            "openai>=1.0.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="langchain memory ai llm recallbricks chatbot conversation",
    project_urls={
        "Bug Reports": "https://github.com/recallbricks/recallbricks-langchain/issues",
        "Source": "https://github.com/recallbricks/recallbricks-langchain",
        "Documentation": "https://recallbricks.com/docs",
    },
)
