"""
In this example, we will visualise two isosurface of the spin density of NiO.
This is also an example of a cell which does NOT have orthogonal lattice vectors
but do not worry, this is taken care of under the hood for you.

The goal here is to see two isosurfaces at a high value of opposite sign on the
Ni atoms, corresponding to spin-up and spin-down as expected from antiferromagnetic
ordering.

Example by: Visagan Ravindran
"""
import numpy as np
import pyvista as pv

import castepfmtvis.plot as plt
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData

# Read the unit cell from a CASTEP .cell file
cell = UnitCell('NiO.cell')

# Read the CASTEP density file.
# NB: CASTEP uses atomic units for the spin density. This involves a factor of 2 to get it in units of hbar/2.
den = GridData('NiO.den_fmt')
spin_sum = np.sum(den.spin)/den.npts
abs_spin_sum = np.sum(np.abs(den.spin))/den.npts
print(f'Integrated spin density:   {spin_sum/2:.7f} hbar/2')
print(f'Integrated |spin density|: {abs_spin_sum/2:.7} hbar/2')

# Initialise a Plotter instance.
plotter = pv.Plotter()

# Make unit cell and add ions
plt.make_cell(plotter, cell)
plt.add_ions(plotter, cell)

# Set the spin density to be the active plot
den.set_current_data(den.spin/den.npts)

# Tip: Use the plt.interactive_isosurface from example 1_interactive_iso to pick good isovalues.
isovalues = [-5e-5, 5e-5]
colors = ['red', 'blue']
labels = ['spin down', 'spin up']
names = ['spin_down', 'spin_up']
plt.plot_isosurface(plotter, den, names, isovalues,
                    colors, labels=labels)

# Add a legend
plotter.add_legend(bcolor='white',
                   size=(0.18, 0.18),  # size of legend as portion of figure
                   border=True,
                   face=None,
                   name='atoms_legend')


# Display figure
plotter.show()
