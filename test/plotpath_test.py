import castepfmtvis.pathplot as pltpath
import castepfmtvis.plot as plt3d
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from castepfmtvis import utils
from castepfmtvis.arithmetic import parse_arithmetic
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData

# Read the grid data
griddata = GridData('test.den_fmt')
fine_grid = griddata.fine_grid


def parse_pt(ptstr: str):
    """Parse fractional coordinates."""
    vec = np.empty(3, dtype=float)
    for i, x in enumerate(ptstr.split()):
        vec[i] = parse_arithmetic(x)
    return vec


def gen_path(startpt: str, direction: str, endpt: str):
    """Generate the expected path from startpt and endpt."""
    direction = direction.upper()

    start_frac = parse_pt(startpt)
    end_frac = parse_pt(endpt)

    gridstart = utils.frac_to_grid_coords(start_frac, fine_grid)
    gridend = utils.frac_to_grid_coords(end_frac, fine_grid)

    gridstart = utils.reduce_grid_pts(gridstart, fine_grid)
    gridend = utils.reduce_grid_pts(gridend, fine_grid)
    # Generate intervals for path pts
    step = 1
    if direction in ('-A', '-B', '-C', '-AB', '-AC', '-BC', '-D'):
        step = -1
    xs = np.arange(gridstart[0], gridend[0]+step, step, dtype=np.int64)
    ys = np.arange(gridstart[1], gridend[1]+step, step, dtype=np.int64)
    zs = np.arange(gridstart[2], gridend[2]+step, step, dtype=np.int64)

    # Get path points
    npoints = xs.shape[0]
    pathpts = np.empty((npoints, 3), dtype=np.int64)
    for i in range(npoints):
        if direction in ('A', '-A'):
            pathpts[i, :] = np.array([xs[i], gridstart[1], gridstart[2]], dtype=np.int64)
        elif direction in ('B', '-B'):
            pathpts[i, :] = np.array([gridstart[0], ys[i], gridstart[2]], dtype=np.int64)
        elif direction in ('C', '-C'):
            pathpts[i, :] = np.array([gridstart[0], gridstart[1], zs[i]], dtype=np.int64)
        elif direction in ('AB', '-AB'):
            pathpts[i, :] = np.array([xs[i], ys[i], gridstart[2]], dtype=np.int64)
        elif direction in ('AC', '-AC'):
            pathpts[i, :] = np.array([xs[i], gridstart[1], zs[i]], dtype=np.int64)
        elif direction in ('BC', '-BC'):
            pathpts[i, :] = np.array([gridstart[0], ys[i], zs[i]], dtype=np.int64)
        elif direction in ('D', '-D'):
            pathpts[i, :] = np.array([xs[i], ys[i], zs[i]], dtype=np.int64)
        else:
            raise AssertionError('Unknown direction')

    return pathpts


# Test all directions
startpts = ['0 0 0', '0 0 0', '0 0 0',
            '1/4 0 0', '0 1/4 0', '0 0 1/4',
            '1/4 1/4 0', '1/4 0 1/4', '0 1/4 1/4',
            '1/4 1/4 0', '1/4 0 1/4', '0 1/4 1/4',
            '1/4 1/4 1/4',
            '3/4 3/4 3/4'
            ]
endpts = ['1/2 0 0', '0 1/2 0', '0 0 1/2',
          '0 0 0', '0 0 0', '0 0 0',
          '1/2 1/2 0', '1/2 0 1/2', '0 1/2 1/2',
          '0 0 0', '0 0 0', '0 0 0',
          '3/4 3/4 3/4',
          '1/4 1/4 1/4'
          ]
directions = ['A', 'B', 'C',  # along cell edges
              '-A', '-B', '-C',  # reverse cell edges
              'AB', 'AC', 'BC',  # along faces
              '-AB', '-AC', '-BC',  # reverse cell face
              'D',  # diagonal
              '-D'  # reverse diagonal
              ]
expect_npts = np.array([fine_grid[0]//2, fine_grid[1]//2, fine_grid[2]//2,
                        fine_grid[0]//4, fine_grid[1]//4, fine_grid[2]//4,
                        fine_grid[0]//4, fine_grid[1]//4, fine_grid[2]//4,
                        fine_grid[0]//4, fine_grid[1]//4, fine_grid[2]//4,
                        fine_grid[0]//2,
                        fine_grid[1]//2], dtype=np.int64)
expect_npts += 1  # CASTEP grid index is Fortran starting from 1.
assert len(startpts) == len(endpts)
assert len(directions) == expect_npts.shape[0]
assert len(startpts) == len(directions)

i = 0
for start, direction, end in zip(startpts, directions, endpts):
    pathspec = [start, direction, end]
    gridpath = pltpath.PathData(griddata, pathspec)

    # Check we have correct number of points in the path each time
    if gridpath.npts != expect_npts[i]:
        print('\nERROR STACK:')
        print('FFT fine grid dimensions=', *fine_grid)
        print(f'{start=} {direction=} {end=}')
        print(f'Points in path: {gridpath.npts=} {expect_npts[i]=}')
        raise AssertionError('Have incorrect number of points in path')

    # Get expected pathpts
    expect_path = gen_path(start, direction, end)
    if expect_path.shape[0] == 0:
        print(f'{start=} {direction=} {end=}')
        raise AssertionError('Empty expected path')

    i += 1

print('SUCCESS: All directions seem to work\n')

# Try reversing path and see if periodic boundary conditions work.
print('CHECKING periodic boundary condtions')
pathspec = ['0 0 0', '-D', '3/4 3/4 3/4']
expect_npts = fine_grid[0]//4 + 1
gridpath = pltpath.PathData(griddata, pathspec)
gridpath.print_path()
assert expect_npts == gridpath.npts

expect_path = np.array([[1,  1,  1],
                        [0,  0,  0],
                        [35, 35, 35],
                        [34, 34, 34],
                        [33, 33, 33],
                        [32, 32, 32],
                        [31, 31, 31],
                        [30, 30, 30],
                        [29, 29, 29],
                        [28, 28, 28]], dtype=np.int64)

assert np.all(gridpath.path_pts == expect_path)
print('SUCCESS: Periodic boundary conditions seem to work')


########################
# Actual Plotting
########################
with open('test.path', 'r', encoding='ascii') as f:
    pathspec = f.readlines()

pathdata = pltpath.PathData(griddata, pathspec)
fig, ax = plt.subplots()
ax.set_ylabel(r'density $\rho(\mathbf{r})$')
ax.set_xlabel(r'position $\mathbf{r}$ along path')

# Format the axis ticks for positions and labels
pathdata.format_pos_ticks(ax)

# Plot the slice
pathdata.plot_slice(ax)
plt.tight_layout()

# Visualise the unit cell
cell = UnitCell('test.cell')
plotter = pv.Plotter()
plt3d.make_cell(plotter, cell)
plt3d.add_ions(plotter, cell)

# Now add arrows to the cell visualisation
pathdata.add_path_arrows(plotter, cell)

# KLUDGE Matplotlib will block execution preventing pyvista from showing the plot
# Unfortunately, when pyvista becomes interactive, the matplotlib window no longer will be interactive
plt.pause(1)
plt.ion()
plotter.show()
