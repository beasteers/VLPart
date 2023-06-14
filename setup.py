import setuptools

setuptools.setup(
    name='vlpart',
    version='0.0.1',
    description='Going Denser with Open-Vocabulary Part Segmentation',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=[
        'torch', 'torchvision', 'supervision',
        'opencv-python', 'timm',
        'clip @ git+https://github.com/openai/CLIP.git@main#egg=clip',
        'detectron2 @ git+https://github.com/facebookresearch/detectron2.git@main#egg=detectron2',
    ],
    extras_require={})
