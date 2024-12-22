"""
In this example, we will visualise the unit cell of silicon
and plot an interactive isosurface of the charge density from a CASTEP calculation.
For a good choice of an isovalue, we expect to see 4 covalent bonds.

Example by: Visagan Ravindran
"""
import castepfmtvis.plot as plt
import numpy as np
import pyvista as pv
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData

# Read the unit cell from a CASTEP .cell file
cell = UnitCell('Si.cell')

# Read the CASTEP charge density.
den = GridData('Si.den_fmt')

# At this stage, one normally sets the object to be plotted.
# On a initialisation of a GridData class, the default object to be plotted will be:
# * For a density (as we have here), it will be the charge density.
# * For a potential, it will be the 1st column for the spin potential in the file
#   (spin 'up' for collinear, V^upup for non-collinear)
# Note the units used by CASTEP, densities are scaled by their respective units multiplied by the number of grid points.
# In other words, sum(charge)/ngridpts = no. of electrons.
# Since we have a density here, this step is superfluous
print(f'No. of electrons= {np.sum(den.charge)/den.npts:.2f}')

# Initialise a Plotter instance.
plotter = pv.Plotter()

# Make unit cell and add ions
plt.make_cell(plotter, cell)
plt.add_ions(plotter, cell, color_scheme='jmol')

# Create an interactive isosurface
plt.interactive_isosurface(plotter, den,
                           label='charge(electrons)',  # label to use on slider
                           color='blue', opacity=0.6)

# Add a legend
plotter.add_legend(bcolor='white',
                   size=(0.20, 0.20),  # size of legend as portion of figure
                   border=True,
                   face=None,
                   name='atoms_legend')

# Display figure
plotter.show()
