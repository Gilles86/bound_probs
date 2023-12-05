from setuptools import setup, find_packages

setup(
    name='bound_probs',
    version='1.0',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
    ],
    include_package_data=True,
    packages=find_packages(include=['bound_probs', 'bound_probs.*']),
)
