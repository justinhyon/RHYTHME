import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="teamflow_placeholder_name",
    version="0.0.1",
    author="Justin Hyon",
    author_email="justinhyon@gmail.com",
    description="package for inter brain realtime EEG experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/justinhyon/teamflow",
    project_urls={},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3-Alpha",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)