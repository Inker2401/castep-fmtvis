"""
This module contains some common routines (such as coordinate conversions)
needed for all modules placed here to avoid circular imports.

In particualar, it s very important that the no castepfmtdata functions are
used as this module can have no dependencies within the program itself.

Author: Visagan Ravindran
"""
from fractions import Fraction

import numpy as np
import numpy.typing as npt
import pyvista as pv
from vtk import vtkMatrix4x4, vtkTransform

#############################
# Real space conversions
#############################


def cart_to_frac(recip_lat: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]):
    """Convert from position in Cartesian coordinates to fractional.

    NB: The reciprocal lattice vectors in recip_lat must be row-ordered.
    """
    # Remember that recip_lat * real_lat = 2pi so make sure to divide by 2pi.
    return np.matmul(recip_lat.T, pos.T)/(2.0*np.pi)


def frac_to_cart(real_lat: npt.NDArray[np.float64], pos: npt.NDArray[np.float64]):
    """Convert from position fractional coordinates to Cartesian.

    NB: The real lattice vectors in real_lat must be row-ordered.
    """
    return np.matmul(real_lat.T, pos.T)


def griddata_to_cart(griddata: npt.NDArray[np.float64],
                     real_lat: npt.NDArray[np.float64]) -> tuple[pv.StructuredGrid, npt.NDArray[np.float64]]:
    """Convert griddata on CASTEP FFT grid to Cartesian space.

    Parameters
    ----------
    griddata : npt.NDArray[np.float64]
        Grid data on FFT fine grid (shape: fine_grid)
    real_lat : npt.NDArray[np.float64]
        Row-ordered real lattice vectors.

    Returns
    -------
    cartgrid: : pv.StructuredGrid
        The corresponding grid in Cartesian space.
    values : npt.NDArray[np.float64]
        The values contained within griddata as a unravelled 1D array.
    """

    # Obtain values from the grid and its dimensions (must be Fortran-ordered!)
    values = griddata.ravel(order='F')
    nx, ny, nz = griddata.shape

    # Create grid dimensions in fractional coordinates
    xs = np.linspace(0, 1, nx, endpoint=True)
    ys = np.linspace(0, 1, ny, endpoint=True)
    zs = np.linspace(0, 1, nz, endpoint=True)

    # Transform each point from fractional coordinates to real space coordinates.
    cartpoints = np.zeros((nx*ny*nz, 3))
    i = 0
    for z in zs:
        for y in ys:
            for x in xs:
                pos = np.array([x, y, z], dtype=np.float64)
                pt = frac_to_cart(real_lat, pos)
                cartpoints[i, :] = pt
                i += 1

    # Now create the 'grid' for plotting within Pyvista in Cartesian space
    # Note the values cannot currently be stored in StructuredGrid as of Pyvista 0.38.5.
    cartgrid = pv.StructuredGrid()
    cartgrid.points = cartpoints
    cartgrid.dimensions = np.array([nx, ny, nz], dtype=np.int64)

    return cartgrid, values


def reduce_frac_pts(frac_pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Imposes periodic boundary conditions on a point in fractional coordinates.

    This ensures that each component of the fractional coordinates x is 0<=x<1.

    Parameters
    ----------
    frac_pos : npt.NDArray[np.float64]
        Point in fractional coordinates.

    Returns
    -------
    npt.NDArray[np.int64]
        Point in fractional coordinates.
    """
    return np.mod(frac_pos, 1.0)

#############################
# Grid coordinate conversions
#############################
# NOTE: The following are depracted and should not be needed.


def frac_to_grid_coords(frac_coords: npt.NDArray[np.float64],
                        fine_grid: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Convert fractional coordinates to FFT fine grid coordinates.

    Figure coordinates start from 1 all the way to the length of each FFT grid dimension.

    NB: In some cases, there may not be a clean mapping.
    For examplem, a fractional coordinate of 1/2 along the 'c'/z-axis of length 9
    would be 4.5 in figure coordinates.
    Therefore, we round up.

    Parameters
    ----------
    frac_coords : npt.NDArray[np.float64]
        Position in fractional coordinates in unit cell.
    fine_grid : npt.NDArray[np.int64]
        Dimensions of the fine grid.

    Returns
    --------
    grid_coords: npt.NDArray[np.int64]
        Positions of atom along FFT fine grid of local potential

    """
    # Calculate grid coordinates noting that they start from 1
    grid_coords = frac_coords * fine_grid + 1

    # Round up grid coordinates so they map onto integers
    grid_coords = np.round(grid_coords).astype(np.int64)
    return grid_coords


def grid_to_frac_coords(grid_coords: npt.NDArray[np.float64],
                        fine_grid: npt.NDArray[np.int64]) -> npt.NDArray[np.int32]:
    """Convert FFT fine grid coordinates to fractional coordinates.

    Figure coordinates start from 1 all the way to the length of each FFT grid dimension.

    Parameters
    ----------
    grid_coords: npt.NDArray[np.int64]
        positions of atom along FFT fine grid of local potential
    fine_grid : npt.NDArray[np.int64]
        Dimensions of the fine grid.

    Returns
    --------
    frac_coords : npt.NDArray[np.float64]
        Position in fractional coordinates in unit cell.
    """
    # Calculate the fractional coordinates
    # Note that grid coordinates run from 1 to ngr_fine in each dimension
    # but fractional coordinates run from 0 to 1.
    frac_coords = (grid_coords - 1)/fine_grid
    return frac_coords


def reduce_grid_pts(pts_grid: npt.NDArray[np.int64] | npt.NDArray[np.int32],
                    fine_grid: npt.NDArray[np.int64]
                    ) -> npt.NDArray[np.int64]:
    """Imposes periodic boundary conditions on a given point.

    NOTE: This assumes that the input data is in grid coordinates rather than
    fractional coordinates.
    Therfore, frac_to_grid_coords should be called first.

    Parameters
    ----------
    pts_grid : npt.NDArray[int] | npt.NDArray[np.int64]
        Point in grid coordinates.
    fine_grid : npt.NDArray[np.int64]
        Dimensions of the FFT grid.

    Returns
    -------
    npt.NDArray[np.int64]
        Periodically equivalent point in grid coordinates.
    """
    return np.mod(pts_grid, fine_grid).astype(np.int64)


######################
# Miscellaneous
######################


def format_fraction(grid_pos: npt.NDArray[np.int64], fine_grid: npt.NDArray[np.int32]) -> str:
    """Format a grid point coordinate as nice fraction where possible.

    If the fraction does not cleanly map on, then just return as decimals with lower precision.

    Parameters
    ----------
    grid_pos : npt.NDArray[np.int64]
        grid coordinates (starting from 1 a la Fortran)
    fine_grid : npt.NDArray[np.int64]
        dimensions of the grid, length = 3

    Returns
    -------
    fmt_frac : str
        formatted string representing grid point in fractional coordinates.

    """

    # Convert from grid coordinates to fractional coordinates
    frac_coord = grid_to_frac_coords(grid_pos, fine_grid)

    fmt_frac = ''
    # Loop over each direction in the coordinate and convert to a pretty fraction.
    for d in range(3):
        frac = Fraction(frac_coord[d])
        # Format the string
        if len(str(frac.numerator)) > 2 or len(str(frac.denominator)) > 2:
            # Fraction is probably something weird,
            # just format to a few decimal places
            fmt_frac += f'{frac_coord[d]:.3f} '
        elif frac.numerator == 0:
            fmt_frac += '0 '
        else:
            fmt_frac += r'$\frac{'+str(frac.numerator)+r'}' + \
                r'{'+str(frac.denominator)+r'}$ '

    return fmt_frac


def make_axes_widget(plotter, real_lat):
    """Add an axes widget indicating the crystallographic directions to the plot.

    Parameters
    ----------
    plotter : PyVista plotter
        Plotter instance to add axes widget.
    real_lat : npt.NDArray[np.float64]
        (row-ordered) real lattice vectors

    Returns
    --------
    axes_widget: vtk.vtkAxesActor
        The axes widget indicator for the plot.
    """

    # Create the axes widget (the arrows will be oriented along the Cartesian rather than crystallographic directions)
    axes_widget = plotter.add_axes(line_width=5,
                                   shaft_length=0.75, tip_length=0.25, cone_radius=0.6,
                                   label_size=(0.25, 0.15),
                                   xlabel='a', ylabel='b', zlabel='c')

    # We perform a coordinate transformation using matrix formed of normalised lattice vectors.
    transform_mat = vtkMatrix4x4()
    for i, lat_vec in enumerate(real_lat):
        norm_vec = lat_vec/np.linalg.norm(lat_vec)
        for j, component in enumerate(norm_vec):
            transform_mat.SetElement(i, j, component)

    # Now apply the transformation matrix to the axes_widget
    transform = vtkTransform()
    transform.SetMatrix(transform_mat)
    axes_widget.SetUserTransform(transform)

    # We now have the correctly oriented axes_widget, just need to tweak the positions.
    label_pos = 1.25
    axes_widget.SetNormalizedLabelPosition(label_pos, label_pos, label_pos)

    return axes_widget
