from setuptools import setup, find_packages

setup(
    name="gomokuagent",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "asyncio",
    ],
    author="sp",
    description="Custom Gomoku AI Agent",
)
