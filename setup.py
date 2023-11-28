from setuptools import setup, find_packages

setup(
    name='music_emotion_classification',
    version='1.0.0',
    author='Valentyn Korniienko',
    author_email='valentyn.korniienko.mknssh.2022@lpnu.ua',
    description='Music Emotion Classification Project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://yourprojectrepository.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'librosa',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'imblearn'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
)
