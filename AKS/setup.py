from setuptools import setup


setup(
    name="aks-sampling-core",
    version="0.1.0",
    description="Adaptive Keyframe Sampling core",
    py_modules=["aks_core"],
    install_requires=["numpy>=1.23"],
)
