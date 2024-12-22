# CASTEP Formatted Visualiser AKA CASTEP-fmtVIS

A visualisation tool for formatted grid data produced by CASTEP.

This tool has been greatly refactored recently for increased modularity to function now more as a library rather than
a single script, thereby providing greater flexibility.
In the future, a Python script serving as an entry-point to the library will be shipped allowing similar functionality
to the previous largely command-line script.

## Installation
The required dependencies are as follows:
- `python>=3.10`
- `numpy>=1.26`
- `pyvista>=0.44.0`
-  vtk>=9.4.0
- `matplotlib`
- `scipy`

Installing via Pip (see below) will automatically install the dependencies.

It is **strongly recommended** that you create a Python virtual environment (and activate it) before
running the install command.

Clone the repository using
```git clone https://gitlab.com/Inker2401/castep-fmtvis.git castep-fmtvis```
This will create a folder called `castep-fmtvis`.

```pip install castep-fmtvis```

Alternatively, you can directly from the official Git project repository by using the command
```pip install git+https://gitlab.com/Inker2401/castep-fmtvis.git```
or you may install from a compressed tarball (`.tar.gz`) using the command (for example)
`pip install castep_fmtvis.tar.gz`.

## Usage
Before anything else, please ensure you can run all Python scripts within the `test` directory.
This is primarily designed to catch changes in PyVista's API between versions that may break the script.

Examples of Python scripts for typical use-cases are provided in the `examples` directory.

## Usage with Other Codes
### Atom Visualisation
The data for visualising the simulation cell is done by the `castepfmtvis.celldata.UnitCell` class.
For ease of use, this can be initialised from a CASTEP `.cell` file but also may be initialised directly as follows:
```
from castepfmtvis.celldata import UnitCell
# Some stuff
cell = UnitCell(real_lat=real_lat, species=species, frac_pos=frac_pos)
```
Here:
- `real_lat` is the real lattice vectors as a 3x3 matrix which each vector as a row (first-index),
- `species` is the list of chemical species of each 'atom' present in a cell.
   You can also specify this to be something that is not a real chemical element to visualise other spheres within the cell for more sophisticated plots alongside the atoms
   or piggy back of the existing element styles.
-  `frac_pos` - the position of each 'atom' in the cell. The shape of this array should be `(len(nspecies),3)`.

### Formatted Data / GridData specification
Although the library was initially designed for CASTEP, it is in principle capable of visualising any 3D on a grid.
You merely need to mimic the CASTEP file format before reading the data into the program
```
    BEGIN header

    Real Lattice(A)               Lattice parameters(A)    Cell Angles
    <a1x> <a1y> <a1z>     a =    <|a1|>  alpha = <alpha>
    <a2x> <a2y> <a2z>     b =    <|a2|>  beta  = <beta>
    <a3x> <a3y> <a3z>     c =    <|a3|>  gamma = <gamma>

    1      F                         ! nspins, non-collinear spin
    <ngx> <ngy> <ngz>                ! fine FFT grid along <a,b,c>
    END header: data is "<a b c> charge" in units of electrons/grid_point * number of grid_points

    1 1 1 <val1>
    .
    <igx> <igy> <igz> <vali>
    .
    ngx ngy ngz <valn>
```
This will initialise the data in the `castepfmtvis.fmtdata.GridData.charge`.
As by default, the current data to be plotted will be divided by the number of grid points to ensure correct normalisation of the density,
you will need to call `castepfmtvis.fmtdata.GridData.set_current_data` to avoid doing this before making a plot.

## Contributing
The easiest way to contribute to this project is to simply use it and open issues on bugs as they arise.
Suggestions of new functionality or quality-of-life enhancements are also welcome.

If you would like to contribute directly to the project, please consult the guidelines contained within `CONTRIBUTING.md`

### Submitting bug reports
Please include steps to reproduce your issue, the error traceback (where applicable) along with the version of Python and Pyvista.
Try to find the simplest possible case that reproduces the bug (ideally with one of the CASTEP input files in `examples`).
Failing this, please provide your input files.

If you have encountered an installation-related error,
please try installing the package within a standalone Python virtual environment separate to the rest of your main Python environment.
Also, please test against one of the example cases given in `examples` and see if it works as expected.

Please use the following report format (and please provide your input files):

Should you still have issues, please submit a bug report according to the following format:

```
Python Version: <your version of Python>
PyVista Version: <your version of PyVista>
Program Version: <version of CASTEP Formatted Visualiser - either by version number of Git commit hash>

Steps to reproduce bug:
1. Step 1
2. Step 2
.
.
.

Traceback:
<Error traceback given when the program runs>
```

## Citation
If you found this program useful, please consider citing it :slight_smile: (see `CITATION.cff`) alongside the reference for PyVista:
Sullivan and Kaszynski, (2019). "PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)". _Journal of Open Source Software_, **4**(37), 1450, https://doi.org/10.21105/joss.01450


## Acknowledgements
Credit to the [JMol team](https://jmol.sourceforge.net/) for the default colour scheme choice for ions.
The list of JMol colours can be found at: <https://jmol.sourceforge.net/jscolors/>

The CPK colour scheme is originally outlined in
Corey-Pauling-Koltun. Walter L. Koltun (1965), _Space filling atomic units and connectors for molecular models_. [U.S. Patent 3170246](https://patents.google.com/patent/US3170246)

The VESTA colour scheme was taken from the VESTA program
K. Momma and F. Izumi, "VESTA: a three-dimensional visualization system for electronic and structural analysis", _Journal of Applied Crystallography_, **41** 653 (2008)

## License
Copyright (C) 2024 V Ravindran
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
