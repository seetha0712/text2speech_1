from setuptools import setup, find_packages

setup(
    name="indian-tts",
    version="0.1.0",
    description="Custom Indian TTS model with male and female voices",
    author="Indian TTS Project",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "phonemizer>=3.2.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "tensorboard>=2.14.0",
    ],
    entry_points={
        "console_scripts": [
            "indian-tts-train=indian_tts.train:main",
            "indian-tts-infer=indian_tts.inference:main",
            "indian-tts-preprocess=indian_tts.data.preprocess:main",
        ],
    },
)
