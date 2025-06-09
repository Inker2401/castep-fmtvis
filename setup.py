from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r', encoding='UTF-8') as file:
    long_description = file.read()

setup(
    name='castepfmtvis',
    version='3.2.1',
    description='A tool for visualing formatted CASTEP potentials, charge and spin densities.',
    long_description=long_description,
    # Requires setuptools >= 38.6.0
    long_description_content_type='text/markdown',
    author='Visagan Ravindran',
    author_email='visagan.ravindran@durham.ac.uk',
    license='GNU General Public License Version 3',

    # Package Requirements
    python_requires='>=3.10',

    packages=find_packages(),
    include_package_data=True,

    install_requires=[
        'ase',
        'matplotlib',
        'numpy',
        # Demand this version of Pyvista and VTK just to avoid changes in API.
        'pyvista>=0.44.0', 'vtk>=9.4.0',
        'scipy',
        'spglib',
        # wheel needed to avoid using legacy setup.py which is depreciated in pip 0.23.1.
        'wheel'
    ]
)
