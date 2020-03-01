""" Setup for the project """

from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name='albert-guide',
        version='0.0.1',
        license='',
        description='Understanding "A Lite BERT". An Transformer approach for '
                    'learning self-supervised Language Models. (wip) ',

        author='Armin Müller',
        author_email='info@dotarmin.info',
        url='',
        packages=find_packages(),

        install_requires=[
            'tensorflow-gpu'
        ],
        python_requires=', ==1.15.2'

    )
