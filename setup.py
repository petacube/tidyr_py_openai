from setuptools import setup, find_packages
import os

setup(
    name="tidyr_py_openai",  # Replace with your package name
    version="0.1.0",  # Replace with your package version
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A Python package integrating tidyr and OpenAI functionalities.",  # Short description
    url="https://github.com/yourusername/tidyr_py_openai",  # Replace with your repository URL
    packages=find_packages(exclude=["tests", "docs"]),  # Packages to include
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Replace with your license
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify your Python version requirements
    install_requires=[
        "openai>=0.27.0",  # Replace with actual dependencies
        "pandas>=1.3.0",
        # Add other dependencies as needed
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "sphinx>=4.0",
            # Add other development dependencies
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            # Add other documentation dependencies
        ],
    },
    include_package_data=True,  # Include other files specified in MANIFEST.in
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/tidyr_py_openai/issues",
        "Documentation": "https://github.com/yourusername/tidyr_py_openai#readme",
        "Source Code": "https://github.com/yourusername/tidyr_py_openai",
    },
)
