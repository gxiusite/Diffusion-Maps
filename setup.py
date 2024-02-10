from setuptools import setup, find_packages

setup(
    name="diffmaps",
    version="0.0.1",
    author="Gezhi Xiu",
    author_email="gezhixiu@gmail.com",
    description="Clear code of diffusion mapping with fast correlation functions.",
    packages=find_packages(),
    install_requires=["scipy", "tqdm", "numpy", "torch", "pandas"],
)
