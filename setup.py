from setuptools import setup, find_packages

setup(
    name='autodj',
    version='0.1',
    description='Automatic DJ for 11-755/18-797 Final Project',
    url='https://github.com/JosephZheng1998/Automated-DJ',
    author='11-755 Group 2',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'colorlog',
        'Essentia',
        'joblib',
        'librosa',
        'numpy',
        'pyAudio',
        'scikit-learn',
        'scipy',
        'yodel',
    ],
    include_package_data=True,
)
