from setuptools import setup, find_packages


REQUIRED_PACKAGES = ['opencv-python',
                     'numpy',
                     'matplotlib',
                     'tqdm',
                     'Pillow']

setup(
    name='Face-Detector-shu244',
    version='0.1.0',
    author="Shuhao Lai",
    author_email="Shuhaolai18@gmail.com",
    description="Extract faces from images",
    packages=find_packages(include=['face_detector', 'face_detector.*']),
    scripts=['api/CustomModelPrediction.py'],
    install_requires=REQUIRED_PACKAGES
)