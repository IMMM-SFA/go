from setuptools import setup, find_packages


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='go',
    version='0.0.0',
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/go',
    license='BSD2-Clause Simplified',
    author='Jordan Kern, Kostas',
    author_email='jkern, kostas',
    description='Grid operations model',
    include_package_data=True,
    install_requires=get_requirements(),

)
