from setuptools import setup, find_packages

with open('README.md') as readme_file:
    README = readme_file.read()

setup_args = dict(
    name='sfa',
    version='0.0.1',
    description='Stochastic Frontier Analysis',
    long_description_content_type="text/markdown",
    long_description=README,
    license='MIT',
    packages=find_packages(),
    author='Sheng Dai',
    author_email='sheng.dai@utu.fi',
    keywords=['SFA', 'MLE', 'TE'],
    url='https://github.com/ds2010/pySFA',
    download_url='https://pypi.org/project/pysfa/',
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    package_data={'sfa': ['data/*.csv']},
)

install_requires = [
    'numpy>=1.19.2',
    'scipy>=1.5.2',
    'scikit-learn>=1.2.2',
    'math',
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)