import imp
import os
from setuptools import setup, find_packages

version = imp.load_source(
    'mlsql.version', os.path.join('mlsql', 'version.py')).version


def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths


js_files = package_files('mlsql/server/js/build')

setup(
    name='mlsql',
    version=version,
    packages=find_packages(exclude=['tests', 'tests.*']),
    package_data={"mlsql": js_files},
    install_requires=[
        'pyspark==2.3.0',
        'scikit-learn',
        "Keras",
        "tensorflow"
    ],
    zip_safe=False,
    author='allwefantasy@gmail.com',
    description='MLSQL: An AI Platform',
    long_description=open('README.rst').read(),
    license='Apache License 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='ml ai sql',
    url=''
)
