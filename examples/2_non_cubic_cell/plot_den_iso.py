"""
In this example, we will visualise the unit cell of silicon
and plot an interactive isosurface of the charge density from a CASTEP calculation.
For a good choice of an isovalue, we expect to see 4 covalent bonds.

Example by: Visagan Ravindran
"""
import castepfmtvis.plot as plt
import pyvista as pv
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData

# Read the unit cell from a CASTEP .cell file
cell = UnitCell('GaAs.cell')

# Read the CASTEP charge density.
den = GridData('GaAs.den_fmt')

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
