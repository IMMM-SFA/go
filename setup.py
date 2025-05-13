import re
from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


def get_requirements():
    """Return a list of package requirements from the requirements.txt file."""
    with open('requirements.txt') as f:
        return f.read().split()


version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", open('go/__init__.py').read(), re.M).group(1)


setup(
    name='go',
    version=version,
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/go',
    license='BSD-2-Clause',
    author='Jordan Kern',
    author_email='jkern@ncsu.edu',
    description='A grid operations model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    python_requires='>=3.9',
    include_package_data=True,
    install_requires=[
        'PyYAML>=5.4.1',
        'requests>=2.25.1',
        'pyomo>=6.0.1',
        'cloudpickle>=3.0.0',
        'pyarrow>=17.0.0',
        'numpy>=1.26.4,<2',
        'pandas>=2.2.2',
    ],
    extras_require={
        'dev': [
            'build>=0.5.1',
            'nbsphinx>=0.8.6',
            'setuptools>=57.0.0',
            'sphinx>=4.0.2',
            'sphinx-panels>=0.6.0',
            'sphinx-rtd-theme>=0.5.2',
            'twine>=3.4.1',
            'pytest>=6.2.4',
            'pytest-cov>=2.12.1',
        ]
    }
)
