from setuptools import setup, find_packages

setup(
    name="forest_change_fl",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'tensorflow',
        'numpy',
        'opencv-python',
        'matplotlib',
        'scikit-learn'
    ]
)