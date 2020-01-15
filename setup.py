from setuptools import setup, find_packages

setup(
    name='nci_eval',
    version='1.0.1',
    description='Supplementary code and materials for paper ' +
        '"On Model Evaluation under Non-constant Class Imbalance" by Brabec et al.',
    url="https://github.com/CiscoCTA/nci_eval",
    author='Jan Brabec, Tomas Komarek',
    author_email='brabecjan91@gmail.com',
    packages=find_packages(exclude=['notebooks', 'test']),
    include_package_data=True,
    install_requires=['matplotlib>=3.1.0', 'numpy>=1.17.3'],
    tests_require=['pytest>=5.2.2,<6']
)
