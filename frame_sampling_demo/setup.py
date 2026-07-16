from setuptools import find_packages, setup


setup(
    name="video-frame-sampling-demo",
    version="0.1.0",
    description="CLI framework for query-aware video frame sampling",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=["numpy>=1.23", "Pillow>=9.0", "decord>=0.6.0"],
    extras_require={
        "clip": ["torch>=2.0", "transformers>=4.40"],
        "ui": ["gradio>=4.44,<6"],
        "dev": ["pytest>=7"],
    },
    entry_points={"console_scripts": ["frame-sampling=frame_sampling_demo.cli:main"]},
)
