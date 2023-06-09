from setuptools import setup, find_packages

setup(
    name='adappt_trainer',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scikit-learn',
        'mlflow'
    ],entry_points={
        'console_scripts': [
            'ml_pipeline=adappt_trainer.cli:main',
        ],
    },
)
