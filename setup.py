from setuptools import setup, find_namespace_packages


def _read(f) -> bytes:
    """
    Reads in the content of the file.
    :param f: the file to read
    :type f: str
    :return: the content
    :rtype: str
    """
    return open(f, 'rb').read()


setup(
    name="wai_ma",
    description="Python library of 2-dimensional matrix algorithms.",
    long_description=(
        _read('DESCRIPTION.rst') + b'\n' +
        _read('CHANGES.rst')).decode('utf-8'),
    url="https://github.com/waikato-datamining/py-matrix-algorithms",
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
    ],
    license='GNU General Public License version 3.0 (GPLv3)',
    package_dir={
        '': 'src'
    },
    packages=find_namespace_packages(where="src"),
    version="0.0.9",
    author='Peter "fracpete" Reutemann',
    author_email='fracpete@waikato.ac.nz',
    install_requires=[
        "numpy",
        "wai_common",
        "wai_test",
        "scikit-learn"
    ],
    include_package_data=True
)
