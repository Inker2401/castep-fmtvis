"""
In this example, we will visualise the density and potential along a path in BaTiO3.

Sometimes, it can be hard to pick out features from just the surface
and we would like to see the actual data itself.
The analogy here is to band structures and Fermi surfaces, where like in
the former, we will plot the density and the potential
along a (real space) path through the unit cell.

In this example, the path is specified through a file 'BaTiO3.path', whose lines read into a list
although one can specify this list within the script directly if one wishes.

Example by: Visagan Ravindran
"""
import castepfmtvis.plot as plt3d
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData
from castepfmtvis.pathplot import PathData

# Read the cell file
seed = 'BaTiO3'
cell = UnitCell(f'{seed}.cell')

# Read the density and potentials
# NB: Potentials do not need to be scaled by the grid points.
den = GridData(f'{seed}.den_fmt')
pot = GridData(f'{seed}.pot_fmt')

# Get the slice of the potential and density along the path.
# First, we get the path specification.
with open(f'{seed}.path', 'r', encoding='ascii') as f:
    pathspec = f.readlines()

# We then initialise an instance of the PathData class by providing griddata. This provides
# information on the grid to dimensions to the PathData class.
gridpath = PathData(den, pathspec)  # currently set to plot the density

# Create a plotter
pl = pv.Plotter()

# Visualise the unit cell
plt3d.make_cell(pl, cell)
plt3d.add_ions(pl, cell)

# Now add arrows to the cell visualisation
gridpath.add_path_arrows(pl, cell,
                         color='blue',
                         # Play around with these numbers!
                         tip_portion=0.25,
                         shaft_portion=0.75,
                         tip_radius=0.2,
                         shaft_radius=0.07,
                         )

# Now plot the slices
fig1, ax1 = plt.subplots()
ax1.set_xlabel(r'position $\mathbf{r}$ in unit cell')
ax1.set_ylabel(r'charge density $\rho(\mathbf{r})$ (electrons)')
gridpath.plot_slice(ax1)

# While one could have mulitple instances of the PathData class, one can alternatively
# just use the GridData.set_current_slice to change the data to plot similar
# to GridData.set_current_data.
# We'll do this for the potential now.
fig2, ax2 = plt.subplots()
ax2.set_xlabel(r'position $\mathbf{r}$ in unit cell')
ax2.set_ylabel(r'potential $v_s(\mathbf{r})$ (Ha)')
gridpath.set_current_slice(pot.cur_data)
gridpath.plot_slice(ax2)

# Format the x-axis tick labels to show the points in the path (like a band structure).
# Custom user labels will be shown at the top of the plot.
gridpath.format_pos_ticks(ax1)
gridpath.format_pos_ticks(ax2)

# Now visualise the figures.
# KLUDGE Matplotlib will block execution preventing pyvista from showing the plot
# Unfortunately, when pyvista becomes interactive, the matplotlib window no longer will be interactive
plt.pause(1)
plt.ion()
pl.show()
