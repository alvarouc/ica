from setuptools import setup

setup(
    name='ica',
    packages=['ica'],
    description='Independent component analysis with INFOMAX algorithm',
    version='0.4',
    url='https://github.com/alvarouc/ica/',
    author='Alvaro Ulloa',
    author_email='alvarouc@gmail.com',
    download_url='https://github.com/alvarouc/ica/tarball/0.4',
    keywords=['infomax', 'ica', 'independent component analysis'],
    install_requires=[
        'numpy',
        'scipy',
    ],
)
