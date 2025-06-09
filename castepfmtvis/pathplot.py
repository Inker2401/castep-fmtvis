"""
This module contains routines for plotting a slice of the 3D data along a path through the unit cell.

All the quantities related to the path are stored in a class called PathData
initialised from a GridData class.
Note that PathData.cur_slice will determine the data plotted by this routine.

The path specication can be done via a file or passing a list that would
contain the lines of said file.
The format is:
   <frac x1> <frac y1> <frac z1> [optional: label]
   [optional: dir]
   <frac x2> <frac y2> <frac z2> [optional: label]
If dir is provided, then an appropriate path is generated
along the two points automatically along the specified direction.
Otherwise, the point will just be added to the path as is.

Author: Visagan Ravindran
"""
import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import pyvista as pv

import castepfmtvis.arithmetic as arit
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData
from castepfmtvis import utils

# Allowed directions to be used in path specification
_VALID_DIRECTIONS = ('A', 'B', 'C',  # along unit cell vectors
                     'D',  # D for diagonal
                     # Directions along faces
                     'AB', 'AC', 'BC',
                     # Aliases
                     'BA', 'CA', 'CB',
                     # Reverse directions
                     '-A', '-B', '-C',
                     '-D',
                     '-AB', '-AC', '-BC',
                     '-BA', '-CA', '-CB'
                     )

# Maximum steps that can be performed for path generation.
_MAX_INTERPOLATE = 1000


class PathData:
    """
    A slice of 3D formatted data on a specified path (like a band structure).

    Similar to GridData, you must set the Grid.Data.cur_slice attribute
    via GridData.set_current_slice the method prior to any plotting.

    NB: The grid coordinates follow CASTEP/Fortran indexing starting from 1 rather than 0.

    Attributes
    ----------
    fine_grid : npt.NDArray[np.int64]
        FFT grid dimensions
    path_pts : npt.NDArray[np.float64]
        Actual points in path in FFT grid coordinates.
    spec_pts : npt.NDArray[np.float64]
        Specified points to create path in FFT grid coordinates.
    spec_labels : list
        Specified point labels.
    npts : int
        Number of points in actual path.
    nspec : int
        Number of points specfied by user to create path.
    npts : npt.NDArray[np.float64]
        Slice of grid data to plot.
    pathspec : list
        Path specification.

    Methods
    --------
    set_current_slice
        Setup a new slice to be plotted from 3D data.
    set_plot_slice
        Plot the current slice along the path
    format_pos_ticks
        Format the labels on the x-axis along the path on the provided axis.
    get_path_frac
        Get the list of points in the path in fractional coordinates.
    get_spec_frac
        Get the list of specified points for the path in fractional coordinates.
    add_path_arrows
        Add arrows to the cell visualisation to show the path taken.
    print_path
        Print out the path information in a pretty format.
    """

    def __init__(self, griddata: GridData,
                 pathspec: list,
                 ):
        """Initialise the PathData class.

        This class can be initialised from a list provided
        whose elements are the lines of a pathfile specifying the path.

        Parameters
        ----------
        griddata : GridData
            Grid data to plot along a path.
        pathspec : list
            Path specification.
            The format is each line of pathfile as an element in a list.
        """
        # Create the path
        self.pathspec = pathspec
        spec_pts, spec_labels, path_pts = _path_create_path(pathspec,
                                                            griddata.fine_grid)

        self.fine_grid = griddata.fine_grid
        self.spec_pts = spec_pts
        self.spec_labels = spec_labels
        self.path_pts = path_pts

        self.nspec = spec_pts.shape[0]
        self.npts = path_pts.shape[0]

        # Now obtain the initial slice of the griddata along this path
        self.cur_slice = np.empty(self.npts, dtype=np.float64)
        for i, pt in enumerate(self.path_pts):
            self.cur_slice[i] = griddata.cur_data[pt[0]-1, pt[1]-1, pt[2]-1]

    def set_current_slice(self, arr: npt.NDArray[np.float64]):
        """Set the current data to use for plotting.

        IMPORTANT: This method must be set each time a new set is to be plotted.
        Otherwise, the old set will be used.

        Parameters
        ----------
        arr : npt.NDArray[np.float64]
            data set to be plotted.

        Raises
        ------
        AssertionError
            arr does not have shape equal to fine_grid.
        """
        if np.all(arr.shape != self.fine_grid):
            raise AssertionError(
                f'Data array shape {arr.shape} is not equal to fine_grid {self.fine_grid}'
            )
        for i, pt in enumerate(self.path_pts):
            self.cur_slice[i] = arr[pt[0]-1, pt[1]-1, pt[2]-1]

    def plot_slice(self, ax: mpl.axes.Axes, color: str = 'b',
                   linewidth: float = 1.75, linestyle: str = '-',
                   label: str | None = None
                   ):
        """Plot the current slice along the path.

        To plot the correct data, ensure that set_current_slice method is called first.

        Parameters
        ----------
        ax : mpl.axes.Axes
            axes to add plot to
        color : str
            colour to use for line
        linewidth : float
            linewidth on plot
        linestyle : str
            linestyle on plot
        """

        ax.plot(self.cur_slice, color=color, linewidth=linewidth,
                linestyle=linestyle, label=label)

    def print_path(self):
        """Print out the path in fractional coordinates."""
        print('Path Properties:')
        print('----------------')
        print(f'Path consisting of {self.npts} generated from {self.nspec} points')
        print('\nPath consists of points <coords> <label>')
        for i, frac in enumerate(self.get_spec_frac()):
            if self.spec_labels[i].strip().lower() == 'none':
                print(
                    f'Point {i+1}      {frac[0]:7.4f} {frac[1]:7.4f} {frac[2]:7.4f}'
                )
            else:
                print(
                    f'Point {i+1}      {frac[0]: 7.4f} {frac[1]: 7.4f} {frac[2]: 7.4f}'
                    + ' '*7+str(self.spec_labels[i])
                )

        print('\nPoints in actual path')
        for i, frac in enumerate(self.get_path_frac()):
            print(
                f'Point {i+1}      {frac[0]: 7.4f} {frac[1]: 7.4f} {frac[2]: 7.4f}'
            )

    def format_pos_ticks(self, ax: mpl.axes.Axes,
                         xtick_fontsize: int = 12,
                         label_fontsize: int = 10,
                         xtick_rotation: int = 20,
                         label_rotation: int = 0,
                         do_label: str = 'both',
                         ) -> mpl.axes._secondary_axes.SecondaryAxis:
        """Format the labels on the x-axis along the path on the provided axis.

        Parameters
        ----------
        ax : mpl.axes.Axes
            Axes object to which labels are added.
        xtick_fontsize : int
            Font size for tick labels for the positions
        label_fontsize : int
            Font size for labels specified by the user
        xtick_rotation : int
            Angle to rotate tick labels for positions
        label_rotation : int
            Angle to rotate user specified labels
        Returns
        ---------
        ax2 : mpl.axes._secondary_axes.SecondaryAxis
            Secondary axes for user labels
        """

        # 09/06/2025 Allow control on which points are labelled. Useful for subplots.
        if do_label not in ('both', 'frac', 'custom'):
            raise ValueError('do_label must be one of: "both", "frac" or "custom"')

        # Determine the unique points in the user's path
        unique_pts = np.unique(self.spec_pts, axis=0)

        # Find the points in the user's path in the actual path
        indx_all = _find_matching_pt(self.path_pts, unique_pts)

        # Now check if we want to actually label those points and grab secondary axes labels
        indx_all, sec_labels = _do_label(
            indx_all, self.spec_pts, self.spec_labels, unique_pts)

        # Now flatten the array
        indx_flat = []
        for i in indx_all:
            for j in i:
                indx_flat.append(j)

        # Now we need to deal with tick labels.
        # First, deal with fractional coordinates
        for i in range(len(indx_all)):
            # Find out what point we are labelling
            pt = unique_pts[i]
            # Convert to fractional coordinates
            fmt_frac = utils.format_fraction(pt, self.fine_grid)
            if i == 0:
                # Initialise list
                pos_labels = [[fmt_frac]*len(indx_all[i])]
            else:
                pos_labels.append([fmt_frac]*len(indx_all[i]))

        # Now flatten the labels
        labels_flat = []
        for i in pos_labels:
            for j in i:
                labels_flat.append(j)
        assert len(labels_flat) == len(indx_flat)

        # Add fractional coordinate labels
        if do_label in ('frac', 'both'):
            ax.set_xticks(indx_flat)
            ax.set_xticklabels(labels_flat, rotation=xtick_rotation,
                               ha='center', fontsize=xtick_fontsize)

        # Now do the same for the user's custom labels
        labels_flat = []
        for i in sec_labels:
            for j in i:
                labels_flat.append(j)

        # Add user custom labels
        if do_label in ('custom', 'both'):
            ax2 = ax.secondary_xaxis('top')
            ax2.set_xticks(indx_flat)
            ax2.set_xticklabels(labels_flat, rotation=label_rotation,
                                ha='center', fontsize=label_fontsize)

            return ax2

    def get_path_frac(self) -> npt.NDArray[np.float64]:
        """Get the list of points in the path in fractional coordinates.

        Returns
        -------
        frac_pts : npt.NDArray[np.float64]
            points in the path
        """
        frac_pts = np.empty((self.npts, 3), dtype=np.float64)
        for i, pt in enumerate(self.path_pts):
            frac_pts[i] = utils.grid_to_frac_coords(pt, self.fine_grid)
        return frac_pts

    def get_spec_frac(self) -> npt.NDArray[np.float64]:
        """Get the list of points specified in fractional coordinates.

        Returns
        -------
        frac_pts : npt.NDArray[np.float64]
            points in the path
        """
        frac_pts = np.empty((self.nspec, 3), dtype=np.float64)
        for i, pt in enumerate(self.spec_pts):
            frac_pts[i] = utils.grid_to_frac_coords(pt, self.fine_grid)
        return frac_pts

    def add_path_arrows(self, plotter: pv.Plotter, cell: UnitCell,
                        color: str = 'red',
                        tip_portion: float = 0.25,
                        shaft_portion: float = 0.75,
                        tip_radius: float = 0.2,
                        shaft_radius: float = 0.07,
                        ):
        """Add arrows to the cell visualisation to show the path taken.

        Parameters
        ----------
        plotter : pv.Plotter
            Plotter instance containing unit cell visualisation.
        cell : UnitCell
            the unit cell to be visualised.
        color : str
            colour to use for arrows.
        tip_portion : float
            portion of total length to use for arrow tips.
        shaft_portion : float
            portion of total length to use for arrow shaft.
        tip_radius : float
            radius of arrow tips.
        shaft_radius : float
            radius of arrow shaft.
        """
        # Get the path points in Cartesian coordinates
        path_cart = np.empty((self.nspec, 3), dtype=np.float64)
        for i, pt in enumerate(self.get_spec_frac()):
            path_cart[i] = utils.frac_to_cart(cell.real_lat, pt)

        # Get arrows between point in each path
        for i in range(1, self.nspec):
            # Get direction and length for each arrow
            start = path_cart[i-1]
            end = path_cart[i]

            # Updated to use uniform arrow creation 22/05/2025
            arrow = _make_arrow(start, end,
                                tip_portion, shaft_portion,
                                tip_radius, shaft_radius
                                )

            plotter.add_mesh(arrow,
                             name=f'arrow_{i}',
                             color=color,
                             smooth_shading=True
                             )

#################
# Path generation
#################


def _path_create_path(pathspec: list, fine_grid: npt.NDArray[np.int64]) -> \
        tuple[npt.NDArray[np.float64], npt.NDArray[np.int64], list, npt.NDArray[np.int32]]:
    """Create a path through real space within the unit cell.

    The expected format of the data in the pathfile is
    <point 1 in frac coords> <label 1(optional)>
    <direction (optional)>
    <point 2 in frac coords> <label 2(optional)>

    If a direction is specified, then interpolation will be
    performed between the two points along the specified direction.
    Otherwise, the point will just be added to the path as is.

    Parameters
    ----------
    pathspec : list
        User's path
    fine_grid : npt.NDArray[np.int64]
        Dimensions of the FFT grid.

    Returns
    -------
    spec_pts : npt.NDArray[np.int64]
        User specified points for path in FFT grid coordinates.
    spec_labels : list
        User specified labels for the  path.
    path_pts : npt.NDArray[np.int64]
        Actual points in the path in FFT grid coordinates.

    Raises
    ------
    KeyError
        Invalid direction specified.

    """
    def _frac_to_grid_periodic(pt_frac, fine_grid):
        """
        Wrapper to frac_to_grid_coords and reduce_pts_unit_cell functions.

        Converts the kpoints to fractional coordinates and
        then will impose periodic boundary conditions.
        """
        pt_grid = utils.frac_to_grid_coords(pt_frac, fine_grid)
        pt_grid = utils.reduce_grid_pts(pt_grid, fine_grid)
        return pt_grid

    def _check_fracs(frac_list: list):
        """Checks if any fractional coordinates were specified in fractions."""
        if len(frac_list) != 3:
            raise IndexError(
                f'Expected 3 fractional coordinates but have {len(frac_list)}'
            )

        read_fracs = np.empty(3, dtype=float)
        for i, val in enumerate(frac_list):
            read_fracs[i] = arit.parse_arithmetic(val)

        return read_fracs

    # Mask to check which lines are direction lines and which are coordinates
    have_dir = np.full(len(pathspec), False, dtype=bool)

    spec_labels = []
    # Loop around the lines to check for directions
    # might as well do labels while we are at it.
    for i, line in enumerate(pathspec):
        split_line = line.strip().split()

        # The length of the line will tell us what is on it
        if (len(split_line)) == 1:
            # Have a direction
            have_dir[i] = True
        elif (len(split_line)) >= 4:
            # Fractional coordinates + a label
            spec_labels.append(' '.join(split_line[3:]).strip())
        else:
            # Fractional coordinates only
            spec_labels.append('')

    # Find the number of points the user specified
    nspec = len(pathspec) - np.count_nonzero(have_dir)
    spec_pts = np.empty((nspec, 3), dtype=np.int64)  # in grid coordinates
    spec_pts_frac = np.empty((nspec, 3))  # in fractional coordinates

    # Set the first point in the path file
    current_pt = _check_fracs(pathspec[0].strip().split()[0:3])
    spec_pts_frac[0] = current_pt

    # Convert to grid coordinates and impose periodic boundary conditions
    current_pt = _frac_to_grid_periodic(current_pt, fine_grid)
    spec_pts[0] = current_pt
    path_pts = spec_pts[0]

    spec_direct = False  # flag stating user desires path between two points
    direct = ''

    # Now run through the path specification and create the interpolated path
    npts = 1
    for i, line in enumerate(pathspec):
        if i == 0:
            # Already done the first line
            continue

        split_line = line.strip().split()

        # Check for direction
        if have_dir[i]:
            direct = split_line[0].strip().upper()
            if direct not in _VALID_DIRECTIONS:
                raise KeyError(
                    f'Invalid direction {direct} specified.'
                )
            spec_direct = True

        else:  # Have a point

            # Get the current point
            current_pt = _check_fracs(split_line[0:3])
            spec_pts_frac[npts] = current_pt
            # Convert to grid coordinates and impose periodic boundary conditions
            current_pt = _frac_to_grid_periodic(current_pt, fine_grid)
            spec_pts[npts] = current_pt

            if spec_direct is True:
                pts = _interpolate_path(
                    spec_pts[npts-1], current_pt, direct,
                    fine_grid)
                # NB: interpolate_path returns the start and end point for interpolation so we would
                # increase the previous point twice.
                # Therefore slice it out (but include the final point since we do not store it)
                pts = pts[1:]
                path_pts = np.vstack((path_pts, pts))
            else:
                # Have just a point manually specified without need for interpolation
                path_pts = np.vstack((path_pts, current_pt))

            # Reset the direction flag
            spec_direct = False
            direct = ''

            # Increase point counter
            npts += 1

    return spec_pts, spec_labels, path_pts


def _interpolate_path(pt1: npt.NDArray[np.int64], pt2: npt.NDArray[np.int32],
                      direct: str, fine_grid: npt.NDArray[np.int64]):
    """Create a path between two grid points along a direction.

    This is done merely for user convenience, no actual interpolation is done
    (if you want a finer grid, use a higher cutoff or increase your grid size in CASTEP!

    Parameters
    ----------
    pt1 : npt.NDArray[np.int64]
        first point in grid coordinates
    pt2 : npt.NDArray[np.int64]
        second point in grid coordinates
    direct : str
        direction to take between two points.
        Specified with respect to crystallographic axes.
    fine_grid : npt.NDArray[np.int64]
        Dimensions of the FFT grid.

    Returns
    -------
    np.array(shape(npts,3))
        points between the two poitns on a path excluding the starting point.

    Raises
    ------
    Exception
        Too many iterations performed, likely path does not exist and user made an error.
    KeyError
        Invalid direction specified.

    """
    # Turn direction to upper case.
    direct = direct.upper()

    # Impose periodic boundary conditions
    # Should already have periodic boundary conditions but let's be sure
    pt1 = utils.reduce_grid_pts(pt1, fine_grid)
    pt2 = utils.reduce_grid_pts(pt2, fine_grid)

    path = np.array(pt1, dtype=np.int64)
    # Now perform interpolation on the grid based on the direction
    current_pt = pt1
    npts = 0  # number of interpolation points

    while not (current_pt == pt2).all():
        # NB: This loop should eventually terminate
        # since we are on the FINE grid and looping around integers...
        if direct == 'A':
            disp = np.array([1, 0, 0], dtype=np.int64)
        elif direct == 'B':
            disp = np.array([0, 1, 0], dtype=np.int64)
        elif direct == 'C':
            disp = np.array([0, 0, 1], dtype=np.int64)
        elif direct == 'D':  # unit cell diagonal
            disp = np.array([1, 1, 1], dtype=np.int64)
        # Along the faces
        elif direct in ('AB', 'BA'):
            disp = np.array([1, 1, 0], dtype=np.int64)
        elif direct in ('AC', 'CA'):
            disp = np.array([1, 0, 1], dtype=np.int64)
        elif direct in ('BC', 'CB'):
            disp = np.array([0, 1, 1], dtype=np.int64)
        # Allow reverse directions
        elif direct == '-A':
            disp = np.array([-1, 0, 0], dtype=np.int64)
        elif direct == '-B':
            disp = np.array([0, -1, 0], dtype=np.int64)
        elif direct == '-C':
            disp = np.array([0, 0, -1], dtype=np.int64)
        elif direct == '-D':
            disp = np.array([-1, -1, -1], dtype=np.int64)
        elif direct in ('-AB', '-BA'):
            disp = np.array([-1, -1, 0], dtype=np.int64)
        elif direct in ('-AC', '-CA'):
            disp = np.array([-1, 0, -1], dtype=np.int64)
        elif direct in ('-BC', '-CB'):
            disp = np.array([0, -1, -1], dtype=np.int64)
        else:
            raise KeyError(f'Invalid direction {direct} specified')

        # Get the next point in path
        current_pt = current_pt + disp
        # Impose periodic boundary conditions
        current_pt = utils.reduce_grid_pts(current_pt, fine_grid)

        # Now add to the path
        path = np.vstack((path, current_pt))
        npts += 1
        if npts >= _MAX_INTERPOLATE:
            # Format points nicely in fractional coordinates
            pt1_frac = utils.format_fraction(utils.grid_to_frac_coords(pt1, fine_grid),
                                             fine_grid)
            pt2_frac = utils.format_fraction(utils.grid_to_frac_coords(pt2, fine_grid),
                                             fine_grid)
            print('\nERROR INFORMATION')
            print('Attempting to find path between following points:')
            print(f'point1={pt1_frac}')
            print(f'point2={pt2_frac}')
            print(f'direction= {direct}')
            raise Exception(
                f'Maximum number interpolation steps {_MAX_INTERPOLATE} have been performed '
                + ' but not yet reached end point. Check your input'
            )

    return path


def _find_matching_pt(path_pts: npt.NDArray[np.float64],
                      unique_pts: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    """Helper function to find where points in the user's path appear in the full path.

    This returns a list of lists. The first index is the index corresponding to the unique_pts index.
    Inside this list index is another list which contains where that unique point appears in the path.
    """
    indx_all = []

    # Loop around the user's unique points and
    # find out where a given point appears in the full path.
    for pt in unique_pts:
        indx = list(np.where(np.all(path_pts == pt, axis=1))[0])
        indx_all.append(indx)

    # Check that they have the same length
    assert len(indx_all) == len(unique_pts)

    return indx_all


def _do_label(indx_all, usr_pts: npt.NDArray[np.float64],
              usr_labels: list,
              unique_pts: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.int64], list]:
    """Helper function to return index of points we want to label."""
    assert len(indx_all) == len(unique_pts)

    # Loop around the unique points
    for i, uniq in enumerate(unique_pts):
        # Get index of actual user points that match the unique point
        indx_for_pt = list(np.where(np.all(uniq == usr_pts, axis=1))[0])

        for j, pnt in enumerate(indx_for_pt):
            # Now get the slice for the labels that correspond to these points
            my_l = usr_labels[pnt].strip()
            if my_l.lower() == 'none':  # We do not want to label the point
                # indx_all already has the points effectively sorted by np.unique so
                # just slice along the first element of indx_for_pt to get where a specific point appears
                # (possibly more than once).
                # Then remove the element based on which one we want
                indx_all[i].pop(j)
            else:
                # Sneak in locating usr_labels since we have to do it anyway
                if i == 0 and j == 0:
                    # Initialise array
                    plot_labels = [[my_l]]
                elif j == 0:
                    # Create extra dimensions
                    plot_labels.append([my_l])
                else:
                    plot_labels[i].append(my_l)

    return indx_all, plot_labels


def _make_arrow(start: npt.NDArray[np.float64], end: npt.NDArray[np.float64],
                tip_portion: float = 0.25,
                shaft_portion: float = 0.75,
                tip_radius: float = 0.2,
                shaft_radius: float = 0.07,
                ):
    """Helper function to make an arrow that stays size consistent.

    This is due to a quirk of pyvista where the scale keyword for arrows
    scales everything, not just the length.
    See documentation of PathData.add_path_arrows method
    for more information on arguments.
    """

    # Get unit vector between both points
    direction = end - start
    length = np.linalg.norm(direction)
    unit_vec = direction / length

    # Get shaft length and tip lengths
    shaft_length = shaft_portion * length
    tip_length = tip_portion * length

    # Construct shaft and tip.
    # NOTE: factor of 2 as Pyvista constructs these
    # from their geometric centre, not the base.
    shaft_center = start + unit_vec * (shaft_length / 2)
    shaft = pv.Cylinder(
        center=shaft_center,
        direction=unit_vec,
        radius=shaft_radius,
        height=shaft_length,
        resolution=150,
    )

    tip_center = start + unit_vec * (shaft_length + tip_length / 2)
    tip = pv.Cone(
        center=tip_center,
        direction=unit_vec,
        height=tip_length,
        radius=tip_radius,
        resolution=125,
    )

    # Combine the two
    arrow = shaft + tip

    return arrow
