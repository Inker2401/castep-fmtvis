"""
This module contains several convenience functions that use Pyvista to plot 3D data on a grid.

Currently, isosurface plots are the only plot types supported
but it should be relatively straightforward to add the others.

Generally, when making a plot within the unit cell, one must:

1. Initialise a Pyvista Plotter instance.
   Generally, this will be the first argument to a lot of functions in this module.
2. The unit cell object should be initialised using the make_cell function.
3. Pass in the grid data to be entered to the plot_isosurface interactive_isosurface functions.
4. The should be created using the create_legend function

Additional atoms can be created using the add_atom_sphere function.

The shapes of the ions are controlled via optional arguments or instead will be taken
from the options set in the ions module.

Author: Visagan Ravindran
"""
import warnings

import numpy as np
import numpy.typing as npt
import pyvista as pv

from castepfmtvis import ion
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData
from castepfmtvis.utils import griddata_to_cart, make_axes_widget

__all__ = ['make_cell', 'add_ions', 'plot_isosurface', 'interactive_isosurface']


def make_cell(plotter: pv.Plotter, cell: UnitCell, boxcolor: str = 'k',
              show_axes: bool = True):
    """Visualise the boundary of a unit cell.

    Parameters
    ----------
    plotter : pv.Plotter
        Pyvista plotter instance to add cell to.
    cell : UnitCell
        unit cell to visualise
    boxcolor : str
        colour of edges to show cell boundary.
    show_axes : bool
        Display a widget showing direction of each lattice vectors with their labels.
    """

    # Add the edges for the unit cell
    # First define the box in fractional coordinates (i.e. a box of 'unit' volume)
    # and then convert to real space/Cartesian coordinates.
    edges = np.array([
        # Bottom face
        [[0, 0, 0], [1, 0, 0]],  # e.g the edge from (0,0,0) to (1,0,0).
        [[0, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0]],
        # Top face
        [[0, 0, 1], [1, 0, 1]],
        [[0, 1, 1], [1, 1, 1]],
        [[1, 0, 1], [1, 1, 1]],
        [[0, 0, 1], [0, 1, 1]],
        # Sides
        [[0, 0, 0], [0, 0, 1]],
        [[1, 1, 0], [1, 1, 1]],
        [[1, 0, 0], [1, 0, 1]],
        [[0, 1, 0], [0, 1, 1]],
    ], dtype=float)

    # Check that we have put all 12 edges of the box.
    assert edges.shape[0] == 12

    for i, edge in enumerate(edges):
        for j, vertex in enumerate(edge):
            edges[i, j] = cell.calc_cart_coord(vertex)

    # DEBUG Colours to make sure I have the edges in the right place.
    # farben = ['g', 'g', 'g', 'g', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b']
    for i, edge in enumerate(edges):
        plotter.add_lines(
            edge, width=3.0, name=f'unit_cell_edge_{i}', color=boxcolor
        )

    # Show directions of each lattice vector with labels if desired.
    if show_axes is True:
        make_axes_widget(plotter, cell.real_lat)


def _add_ions_periodic(plotter: pv.Plotter,
                       frac_pos: npt.NDArray[np.float64],
                       cell: UnitCell,
                       element: str, iatom: int,
                       radius: float, color: str,
                       theta_resolution: int,
                       phi_resolution: int,
                       ):
    """Adds appropriate periodic images for atom at specified position.

    Parameters
    ----------
    plotter : pv.Plotter
        Pyvista plotter instance to add cell to.
    frac_pos : npt.NDArray[np.float64]
        position of ion
    cell : UnitCell
        unit cell to visualise
    element : str
        chemical symbol for element
    iatom : int
        species index
    radius : float
        radius to use for atom sphere
    color : str
        colour to use for atom sphere
    theta_resolution : int
        number of points to use in azimuthal direction
    phi_resolution : int
        number of points to use in longitudinal direction
    """
    def _add_periodic_spheres(fracs_periodic, labels_periodic):
        # Convert the fractional coordinates obtain into Cartesian coordnates
        assert fracs_periodic.ndim == 2
        assert isinstance(labels_periodic, list) is True
        carts_periodic = np.empty(fracs_periodic.shape, dtype=np.float64)
        for i, pos in enumerate(fracs_periodic):
            carts_periodic[i] = cell.calc_cart_coord(pos)

        # Make spheres and add to plot.
        for i, pos in enumerate(carts_periodic):
            ion_sphere = pv.Sphere(radius, pos,
                                   theta_resolution=theta_resolution,
                                   phi_resolution=phi_resolution)
            plotter.add_mesh(ion_sphere, color=color,
                             name=labels_periodic[i],
                             smooth_shading=True
                             )

    # First check if we have any atoms at the corners and add their periodic images.
    fracs_corner, labels_corner = ion.get_periodic_corner(frac_pos,
                                                          iatom, element)
    if fracs_corner is not None:
        _add_periodic_spheres(fracs_corner, labels_corner)

    # Now do the same thing for the faces.
    fracs_faces, labels_faces = ion.get_periodic_face(frac_pos,
                                                      iatom, element)
    if fracs_faces is not None:
        # Need to reshape since _add_periodic_spheres assumes 2D array
        fracs_faces = fracs_faces.reshape(1, 3)
        labels_faces = [labels_faces]
        _add_periodic_spheres(fracs_faces, labels_faces)


def add_ions(plotter: pv.Plotter, cell: UnitCell,
             elements: list | str | None = None,
             frac_pos: npt.NDArray[np.float64] | None = None,
             radius: list | float | None = None,
             atomic_scale: float = 0.15,
             color: list | str | None = None,
             color_scheme: dict | str = 'vesta',
             theta_resolution: int = 80,
             phi_resolution: int = 80,
             labels: list | str | None = None,
             periodic: bool = True
             ):
    """Plot a list of ions within a unit cell.

    By default, if no elements are provided, then the data is taken from the cell
    instead. Otherwise, taken from elements and frac_pos must be set.

    Parameters
    ----------
    plotter : pv.Plotter
        Pyvista plotter instance to add atom spheres.
    cell : UnitCell
        unit cell to use (for lattice vectors)
    elements : list | str | None
        list of chemical species
    frac_pos : npt.NDArray[np.float64]
        atomic positions in fractional coordinates
    radius : list | float | None
        radius to use for spheres
        (if not specified, van der Waals radius will be used scaled by
        atomic_scale)
    atomic_scale : float
        scale to apply to van der Waals radius to use as plotting radius
    color : list | str | None
        colour to use for atomic spheres
    color_scheme : dict | str
        colour scheme to use if colours is None ('jmol', 'cpk' or 'vesta')
        Alternatively, a dictionary defining a custom colour scheme may be defined.
        This must have the elements (only those in the plot, no need for the the full periodic table!)
        as the keys and the actual colours as values.
    theta_resolution : int
        number of points to use in longitudinal direction
    phi_resolution : int
        number of points to use in longitudinal direction
    labels : list | str | None
        labels to use for atoms
        (Defaults: Unique species in the unit cell)
    periodic : bool
        Imposes periodic boundary conditions on atoms in visualisation.

    Returns
    -------

    Raises
    ------
    ValueError
        Invalid color_scheme specified, must be one of 'jmol', 'cpk' or 'vesta'
    AssertionError
        Arguments have the wrong shape.
    """
    # Get ions we want to plot
    if elements is None:
        elements = cell.species
    elif isinstance(elements, list) is False:
        elements = list(elements)

    natoms = len(elements)

    # Get the positions
    if frac_pos is None:
        frac_pos = cell.frac_pos
    else:
        frac_pos = np.array(frac_pos, dtype=np.float64)
        # Turn into 2D array since we loop over coordinates later
        frac_pos = frac_pos.reshape(1, -1)
        if frac_pos.shape[0] != natoms:
            raise AssertionError(
                f'Size of index 1 of frac_pos {frac_pos.shape[0]} ' +
                f'does not match no. of atoms ({natoms})')

    # Check remaining arguments that they are correct.
    def _arg_to_list(arg, label):
        """Turn a float or string arg into filled list with that value"""
        if isinstance(arg, list) is False:
            arg = [arg for i in range(natoms)]
        else:
            if len(arg) != natoms:
                raise AssertionError(
                    f'No. of {label} ({len(arg)}) ' +
                    f'does not match no. of atoms ({natoms})'
                )
        return arg

    def _set_def_param(param: str, element: str, cdict: dict, default_val: str | float):
        try:
            val = cdict[element]
        except KeyError:
            # Element not found
            warnings.warn(f'Unknown element {element} in cell file. '
                          + f'{param} defaulting to {default_val}')
            val = default_val
        return val

    if radius is not None:
        radius = np.array(_arg_to_list(radius, 'radius'), dtype=float)
    else:
        radius = np.array([0.0 for i in range(natoms)], dtype=float)
        for i, sp in enumerate(elements):
            radius[i] = _set_def_param('Radius', sp, ion.VDW_RADIUS, 2.5)
        # Scale by atomic scale
        radius *= atomic_scale

    if color is not None:
        color = _arg_to_list(color, 'colour')
    else:
        # Use CPK or JMOL colours
        color = ['' for i in range(natoms)]
        for i, sp in enumerate(elements):
            if color_scheme == 'jmol':
                color[i] = _set_def_param('Colour', sp,
                                          ion.JMOL_COLOURS, '#FFFFFF')  # white
            elif color_scheme == 'cpk':
                color[i] = _set_def_param('Colour', sp,
                                          ion.CPK_COLOURS, 'FFC0CB')  # pink
            elif color_scheme == 'vesta':
                # Added VESTA colour scheme 21/05/2025
                color[i] = _set_def_param('Colour', sp,
                                          ion.VESTA_COLOURS, '#4C4C4C')  # dark grey
            elif isinstance(color_scheme, dict):
                try:
                    color[i] = color_scheme[sp]
                except KeyError:
                    raise KeyError(f'Missing element {sp} in custom scheme.')
            else:
                raise ValueError('Invalid colour scheme specified, ' +
                                 'must be ("cpk", "jmol", "vesta")')

    if labels is not None:
        labels = _arg_to_list(labels, 'labels')
    else:
        labels = [None for i in range(natoms)]

    ion_meshes = []
    unique_species = []  # elements present within the cell. Needed for legend.
    iatom = 0

    for sp, frac, r, c, label in zip(elements, frac_pos, radius, color, labels):
        # Calculate Cartesian position
        pos = cell.calc_cart_coord(frac)

        # Create sphere for ion
        ion_sphere = pv.Sphere(r, pos,
                               theta_resolution=theta_resolution,
                               phi_resolution=phi_resolution)

        # Add sphere to plot along with a label if necessary.
        # If it is the first one of that element,
        # then add a label for that atom ONLY.
        if label is None:
            if sp not in unique_species:
                unique_species.append(sp)
                use_label = f'{sp}'
            else:
                use_label = None
        else:
            use_label = label

        ion_mesh = plotter.add_mesh(ion_sphere,
                                    color=c,
                                    name=f'{sp}_{iatom}',  # must be unique
                                    label=use_label,
                                    smooth_shading=True
                                    )
        ion_meshes.append(ion_mesh)

        # Add periodic boundary copies of ions, if required
        if periodic is True:
            _add_ions_periodic(plotter, frac, cell, sp, iatom, r, c,
                               theta_resolution, phi_resolution)

        iatom += 1


def plot_isosurface(plotter: pv.Plotter, griddata: GridData,
                    name: list | str,
                    isovalue: list | float,
                    color: list | str = 'blue',
                    opacity: list | float = 0.6,
                    labels: list | str | None = None,
                    iso_method: str = 'contour'
                    ):
    """Plot (an) isosurface(s) for the given set of griddata.

    Make sure you set the current data set using griddata.set_current_data
    before making the plot.

    Parameters
    ----------
    plotter : pv.Plotter
        Pyvista plotter instance for isosurface.
    griddata : GridData
        data on a grid to be visualised.
    name : list | str
        name to use for Pyvista mesh (must be unique each time!)
    isovalue : list | float, optional
        list of isovalue(s) to use for isosurface(s)
    color : list | str
        colour(s) to use for isosurface(s)
    opacity : list | float
        opacity/opacities for each isosurface
    labels : list | str, optional
        label(s) to use for each isosurface
    iso_method : str
        method for creating isosurface ('contour', 'flying_edges', 'marching_cubes')

    Raises
    ------
    AssertionError
        Arguments have the wrong shape.
    """

    # Check that the current set in griddata has been set.
    if griddata.cur_data is None:
        raise AssertionError('griddata has no current set')

    if np.all(griddata.cur_data.shape != griddata.fine_grid):
        raise AssertionError(f'Current set for griddata has shape {griddata.cur_data.shape}' +
                             f'not equal to fine_grid {griddata.fine_grid}')
    if isinstance(isovalue, list) is False:
        isovalue = [isovalue]
    niso = len(isovalue)

    # Transform data from fine grid to Cartesian space.
    grid, values = griddata_to_cart(griddata.cur_data, griddata.real_lat)

    def _proc_args(arg, label):
        """Turn an arg into filled list with that value"""
        if isinstance(arg, list) is False:
            arg = [arg for i in range(niso)]
        else:
            if len(arg) != niso:
                raise AssertionError(
                    f'No. of {label} ({len(arg)}) ' +
                    f'does not match no. of isosurfaces ({niso})'
                )
        return arg

    # Check optional arguments provided and turn to lists.
    color = _proc_args(color, 'color')
    opacity = _proc_args(opacity, 'opacity')
    labels = _proc_args(labels, 'labels')
    assert name is not None
    name = _proc_args(name, 'name')

    # Plot isosurfaces meshes
    for val, c, opa, n, l in zip(isovalue, color, opacity, name, labels):
        iso = grid.contour([val], scalars=values, rng=[0, 5],
                           method=iso_method)
        plotter.add_mesh(iso, name=n,
                         color=c,
                         label=l,
                         opacity=opa,
                         smooth_shading=True
                         )


def interactive_isosurface(plotter: pv.Plotter, griddata: GridData,
                           color: str = 'blue',
                           opacity: float = 0.6,
                           label: str | None = None,
                           name: str = 'isosurface_pot_mesh',
                           slide_rng: list | None = None,
                           iso_method: str = 'contour',
                           pointa: tuple = (0.1, 0.9),
                           pointb: tuple = (0.6, 0.9),
                           isorng: tuple = (0, 5)
                           ):
    """Plot an interactive isosurface to a set of data.

    Make sure you set the current data set using griddata.set_current_data
    before making the plot.

    Note that if you are using this function to make multiple interactive isosurfaces,
    each isosurface must have a different name.

    Parameters
    ----------
    plotter : pv.Plotter
        Pyvista plotter instance for isosurface.
    griddata : GridData
        data on a grid to be visualised.
    isovalue : float
        list of isovalue(s) to use for isosurface(s)
    color : str
        colour to use for isosurface
    opacity : float
        opacity/opacities for each isosurface
    label : str | None
        label to use for slider
    name : str
        name to use for Pyvista mesh (must be unique each time!)
    slide_rng : str | None
        range of values to use for slider By default will be minimum and max value of data.
    iso_method : str
        method for creating isosurface ('contour', 'flying_edges', 'marching_cubes')
    pointa: tuple
        The relative coordinates of the left point of the slider on the display port.
    pointa: tuple
        The relative coordinates of the right point of the slider on the display port.
    rng : tuple
        The range of values to look for contours.
    """
    # Check that the current set in griddata has been set.
    if griddata.cur_data is None:
        raise AssertionError('griddata has no current set')

    if np.all(griddata.cur_data.shape != griddata.fine_grid):
        raise AssertionError(f'Current set for griddata has shape {griddata.cur_data.shape}' +
                             f'not equal to fine_grid {griddata.fine_grid}')

    # Transform data from fine grid to Cartesian space.
    grid, values = griddata_to_cart(griddata.cur_data, griddata.real_lat)

    # Set slider range
    if slide_rng is not None:
        print(slide_rng)
        min_val, max_val = np.sort(slide_rng)
    else:
        min_val, max_val = np.min(values), np.max(values)

    # Set starting value for isosurface
    isovalue = 0.5 * (max_val - min_val)

    def create_isosurface(value):
        """Creates the initial isosurface."""
        # HACK / KLUDGE - needs to be done this way as Pyvista
        # only accepts single arguments for sliders.
        myval = float(value)
        iso = grid.contour([myval], scalars=values, rng=isorng,
                           method=iso_method)

        # create initial plot
        # NB : It needs to be named so it updates correctly
        plotter.add_mesh(iso, name=name,
                         color=color,
                         smooth_shading=True,
                         opacity=opacity,
                         )

    plotter.add_slider_widget(create_isosurface,
                              [min_val, max_val],
                              value=float(isovalue),  # starting value
                              title=label,
                              pointa=pointa,
                              pointb=pointb
                              )
