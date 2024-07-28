from setuptools import setup, find_packages

setup(
    name='registration',
    version='1.0.0',
    author='Atchuth Naveen',
    author_email='achilapa@uci.edu',
    description='Registration package',
    url='https://github.com/UCI-ALLEN-BrainRepositoryCode/registration',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'numpy',
        'scipy',
        'scikit-image',
        'matplotlib',
        'tqdm',
        'joblib',
    ],
    # Include additional files into the package
    include_package_data=True,
)