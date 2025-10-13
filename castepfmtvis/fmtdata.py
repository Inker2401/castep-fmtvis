"""
This module handles the reading of formatted grid data.

The main data is stored in a GridData class instance.
The data to actually be plotted is in the cur_data attribute of the class
set by the set_current_data method.

Although currently programmed for CASTEP formatted grid files
(e.g. .den_fmt or .pot_fmt), in principle any data on a rectilinear grid can be read.
It should be relatively straightforward to extend this to read data from other codes
as a result.
To do this, create a function similar to read_castep_fmt and then add the
relevant attributes.

Author: Visagan Ravindran
"""
import numpy as np
import numpy.typing as npt

from castepfmtvis import io
from typing import Optional, Tuple

__all__ = ['GridData', 'read_castep_fmt', 'read_real_lat_fmt',
           'den_spin_to_rho_up_down', 'rho_up_down_to_den_spin']


def read_real_lat_fmt(filename: str) -> npt.NDArray[np.float64]:
    """Read the lattice vectors from the CASTEP formatted file

    Parameters
    ----------
    filename : str
        unit cell to use

    Returns
    -------
    real_lat: np.ndarray
        lattice vectors (order is each vector is a 'row')
    """

    header = io.extract_header(filename)
    lat_lines = header[3:6]

    real_lat = np.empty((3, 3), dtype=np.float64)
    for i, line in enumerate(lat_lines):
        real_lat[i] = np.array(line.split()[:3], dtype=np.float64)

    return real_lat


def read_castep_fmt(filename: str, is_den: bool,
                    nblank_header: int = 1) -> tuple[list[str], npt.NDArray[np.float64]]:
    """Read a formatted CASTEP data file.

    Note this routine simply reads the data as is and does not do any normalisation, e.g. of the
    density. The spin component index also uses a slightly different convention from CASTEP
    storing the following depending on whether we have a density or potential respectively.
    * 1: spin degenerate (charge and potential)
    * 2: collinear spin (charge/spin and up/down potentials)
    * 3: non-collinear potential (upper diagonal of spin potential matrix)
    * 4: non-collinear density (charge, components of non-collinear spin density)

    Parameters
    ----------
    filename : str
        formatted file to read
    is_den: bool
        Are we reading a density or a potential?
    nblank_header : int
        number of blank lines after header to read

    Returns
    -------
    header : list[str]
        header contents in formatted file
    gridvals : np.ndarray
        array of formatted values (shape=(nspins, nx, ny, nz))
        where nx, ny, nz are the dimensions of the fine grid.

    Raises
    ------
    IOError
        Reached EOF before reading all grid points.
    """
    # Read the header to know what the file contains
    header, start, end = io.extract_header(filename, ret_lineno=True)

    # Extract spin information
    line = header[7]
    if 'non-collinear spin' in line:
        # "New" CASTEP format - allows for non-collinear data so make sure to read the flag.
        nspins, have_nc = line.split()[:2]
        if have_nc == 'T':
            if is_den is True:
                # For density, allocate density storing the charge in the first spin index
                ns = 4
            else:
                # Non-collinear spin being used so have 3 spin components (x,y,z)
                ns = 3
        elif have_nc == 'F':
            # Not using collinear so read CASTEP number of spin components.
            ns = int(nspins)
        else:
            raise ValueError(
                f'Unknown value for non-collinear spin flag: "{have_nc}"'
            )
    else:
        # 'Legacy' format used by old code such as OEP which does not support non-collinear.
        ns = int(line.split()[0])

    # Get the fine grid dimensions.
    nx, ny, nz = np.array(header[8].split()[:3], dtype=np.int64)

    # Now read the file
    gridvals = np.empty((ns, nx, ny, nz))
    with open(filename, 'r', encoding='ascii') as f:
        i = 0  # no of values read
        for n, line in enumerate(f):
            if n < end+nblank_header+1:
                # In CASTEP there is sometimes a blank line (or more) between the
                # header and the actual data so make sure to skip these lines.
                continue

            # Read the current point making sure to include it's grid index
            # as they can may not be in order!
            split_ln = line.split()
            ix, iy, iz = split_ln[:3]
            ix, iy, iz = int(ix)-1, int(iy)-1, int(iz)-1  # Fortran vs C indexing!
            gridvals[:, ix, iy, iz] = np.array(split_ln[3:], dtype=np.float64)

            # Increase number of points read counter
            i += 1

            # Exit if we have correct number of lines. 19/05/2025
            # This is important for reading meta-GGA calculations 19/05/2025
            # as the KED data is simply read at the end.
            if i == nx*ny*nz:
                break

    # Now check that all grid points were actually read
    if i != nx*ny*nz:
        raise IOError('Reached EOF but did not read all grid points')

    return header, gridvals


def den_spin_to_rho_up_down(charge: npt.NDArray[np.float64],
                            spin: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate charge densities for each collinear spin channel from total charge density and spin density.

    Parameters
    ----------
    charge : npt.NDArray[np.float64]
        total charge density on a real-space grid
    spin : npt.NDArray[np.float64]
        spin density on a real-space grid

    Returns
    -------
    rhoup, rhodown : tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        densities for each spin channel on a real-space grid

    Raises
    ------
    AssertionError
        Grid dimensions do not match between charge and spin.

    """

    if charge.shape != spin.shape:
        raise AssertionError(f'Charge array has shape {charge.shape} ' +
                             f'but spin array has {spin.shape}')

    rhoup = (charge + spin)/2.0
    rhodown = (charge - spin)/2.0
    return rhoup, rhodown


def rho_up_down_to_den_spin(rhoup: npt.NDArray[np.float64],
                            rhodown: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate total charge and spin density from charge densities for each collinear spin channel.

    Parameters
    ----------
    rhoup : npt.NDArray[np.float64]
        spin-up density on a real-space grid
    rhodown : npt.NDArray[np.float64]
        spin-down density on a real-space grid

    Returns
    -------
    rhoup, rhodown : tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        charge and spin densities a real-space grid

    Raises
    ------
    AssertionError
        Grid dimensions do not match between rhoup and rhodown.

    """
    if rhoup.shape != rhodown.shape:
        raise AssertionError(f'rhoup array has shape {rhoup.shape} ' +
                             f'but rhodown array has {rhodown.shape}')

    charge = rhoup + rhodown
    spin = rhoup - rhodown
    return charge, spin


class GridData():
    """Grid data from CASTEP to be visualised.

    Although CASTEP by default will write .den_fmt and .pot_fmt files
    for densities and potentials out-of-the-box, this format is actually more general
    and can be used for plotting a variety of grid-based data.

    The data is stored in different arrays depending on the data one wishes to plot.
    - For densities, the data will be allocated in:
        1. charge (shape: fine_grid)
        2. spin for collinear spin (shape: fine_grid)
        3. ncspin for non-collinear spin (shape: (3, fine_grid))
      Each of these is normalised  such that their integral is the sum
      of all points divided by number of grid points.

    - For potentials, the data will be allocated in:
        1. pot (shape: (nspins, fine_grid)) where nspins is 2 for collinear spin-polarisation, 1 otherwise.
        2. ncpot, if using non- collinear (shape: (3, fine_grid)).
           Note CASTEP only stores upper/lower-triangular part of
           the potential matrix since it is Hermitian.

      These are in physical units so the integral is simply the sum of their values.

    If using your own data with same formatted file format as CASTEP,
    depending on your preference, override is_den argument when initialising the class
    and either charge or pot array will be allocated.

    Alternatively you can initialise the class directly by specifying the necessary attributes.
    When done in this manner, only the cur_data array will be initialised (see below).

    There is also an auxiliary array called cur_data, that is needs to be set prior to using any plotting routines.
    This is achieved by using the set_current_data method.
    By default, this will default to the charge/npts or pot[0,:,:,:]
    depending on whether is_den is True or False.

    IMPORTANT: Prior to plotting, the set_current_data method must be used to ensure the right set is plotted.

    Attributes
    -------
    is_den : bool
        Is the data a density or a potential?
    real_lat : str
        Real lattice (mainly for sanity checking input)
    fine_grid: npt.NDArray[np.int64]
        Dimensions of the FFT grid.
    npts: int
        Number of points in FFT grid.
    cur_data: npt.NDArray[np.float64]
        The data that is to be actually plotted set.
        By default, this will default to the charge/den or pot[0,:,:,:] arrays
        depending on whether is_den is True or False.

    nspins: int
        The number of spin channels (1: spin unpolarised, 2: collinear)
    have_nc: bool
        Whether this is a non-collinear calculation or not.
    is_den: bool
        Current file is an electronic density.

    units : str
        Units for the density or potential.
    charge: npt.NDArray[np.float64]
        Charge density (units: electrons*grid_pts) - see however have_rho_up_down
    spin: npt.NDArray[np.float64]
        Spin density (units: spin*grid_pts)  - see however have_rho_up_down
    pot: npt.NDArray[np.float64]
        Potential for each (collinear) spin-channel (units: Hartrees)
    ncpot: npt.NDArray[np.float64]
        Potential for non-collinear spin-potential (units : Hartrees)

    have_rho_up_down : bool
        Flag for whether charge and spin arrays instead contain charge densities
        for each spin channel, i.e. rhoup and rhodown respectively.

    Methods
    --------
    set_current_data
        Set the data array that must be used for plotting.
    get_rho_up_down
        Get the charge densities for each spin channel from charge and spin densities.
    get_charge_spin
        Get the charge and spin densities from charge densities for each spin channel.
    shift_grid_data
        Performs a cyclic shift of grid data given a shift in fractional coordinates.
    """

    def __init__(self, filename: str | None = None,
                 is_den: bool | None = None,
                 nspins: int | None = None,
                 have_nc: bool | None = None,
                 nblank_header: int = 1,
                 real_lat: npt.NDArray[np.float64] | None = None,
                 datarr: npt.NDArray[np.float64] | None = None,
                 charge: npt.NDArray[np.float64] | None = None,
                 spin: npt.NDArray[np.float64] | None = None,
                 ncspin: npt.NDArray[np.float64] | None = None,
                 pot: npt.NDArray[np.float64] | None = None,
                 ncpot: npt.NDArray[np.float64] | None = None,
                 units: str = ''
                 ):
        """
        Initialise rectilinear grid data.
        This can be initialised in two ways:
          1) Read formatted CASTEP rectilinear grid data
          2) Specifying the relevant attributes directly.
             NB: The grid data will be stored in the cur_data class.

        When initialising directly, both real_lat and dataarr must be set.

        Parameters
        ----------
        filename : str
            formatted file to read
        is_den : bool | None
            Is file a density file?
            If not specified, it will be inferred from the file extension
        nblank_header : int
            number of blank lines after header

        All other arguments follow attributes in class definition.
        """
        # Declare empty arrays first and allocate later
        self.charge: npt.NDArray[np.float64] | None = None
        self.spin: npt.NDArray[np.float64] | None = None
        self.ncspin: npt.NDArray[np.float64] | None = None
        self.pot: npt.NDArray[np.float64] | None = None
        self.ncpot: npt.NDArray[np.float64] | None = None
        self.cur_data: npt.NDArray[np.float64] | None = None
        self.have_rho_up_down = False

        # 09/06/2025 Decide how we want to initialise the file
        if filename is not None:
            # Check if we have a density, otherwise look at the file extension
            # The main difference here apart from units is the FFT convention
            # used in CASTEP. Densities are normalised to the electrons * ngridpts
            # whereas potentials are in normal atomic units.
            if is_den is None:
                if filename.endswith('.den_fmt'):
                    self.is_den = True
                else:
                    # Assume potential at least as far as normalisation is concerned
                    self.is_den = False
            else:
                self.is_den = is_den

            # Read the file
            assert self.is_den is not None  # for type checks
            header, gridvals = read_castep_fmt(filename, self.is_den, nblank_header)

            # Extract fine grid dimensions
            self.fine_grid = np.array(header[8].split()[:3], dtype=int)
            self.npts = np.prod(self.fine_grid)

            # Extract the unit cell vectors
            self.real_lat = read_real_lat_fmt(filename)

            # Extract spin information and set non-collinear flag
            nsets = gridvals.shape[0]
            self.have_nc = nsets in (3, 4)
            if self.have_nc is True:
                self.nspins = 1
            else:
                self.nspins = nsets

            # Allocate necessary grid data
            if self.is_den is True:
                # Allocate charge density
                self.charge = gridvals[0, :, :, :]

                # Check if we need to allocate a spin density
                if self.have_nc is True:
                    self.ncspin = gridvals[1:, :, :, :]
                elif self.nspins != 1:
                    self.spin = gridvals[1, :, :, :]

            else:
                # Allocate potentials
                if self.have_nc is True:
                    self.ncpot = gridvals
                else:
                    self.pot = gridvals

            # Set default set to be plotted - this will hopefully be the most modular/flexible 19/05/2025
            if self.is_den is True:
                self.set_current_data(self.charge/self.npts)  # normalisation
                self.units = 'electrons'
            else:
                self.set_current_data(self.pot[0, :, :, :])
                self.units = 'Hartrees'

        else:  # Direct initialisation
            if real_lat is None:
                raise ValueError('real_lat must be provided for direct initialisation')
            if datarr is None:
                raise ValueError('datarr must be provided for direct initialisation')
            if datarr.ndim != 3:
                raise IndexError('datarr must be a 3D array')

            self.real_lat = real_lat
            self.cur_data = datarr

            if len(units) == 0:
                self.units = 'a.u.'
            else:
                self.units = units.strip()

            # Set other necessary info
            self.fine_grid = np.array(self.cur_data.shape, dtype=int)
            self.npts = np.prod(self.fine_grid)

            # Set other (potentially) dummy info
            if is_den is None:
                self.is_den = False
            else:
                self.is_den = is_den

            if nspins is None:
                self.nspins = 1
            else:
                self.nspins = nspins

            if have_nc is None:
                self.have_nc = False
            else:
                self.have_nc = have_nc

            # Set auxiliary arrays that may have been provided 08/10/2025
            if charge is not None:
                self.charge = charge
            if spin is not None:
                self.spin = spin
            if ncspin is not None:
                self.ncpot = ncpot
            if pot is not None:
                self.pot = pot
            if ncpot is not None:
                self.ncpot = ncpot

    def set_current_data(self, arr: npt.NDArray[np.float64]):
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
        self.cur_data = arr

    def get_rho_up_down(self, ret_arrs: bool = False) -> Optional[Tuple[npt.NDArray[np.float64],
                                                                        npt.NDArray[np.float64]]]:
        """Get the charge density for each spin channel.

        By default, the 'up' channel or whatever the first spin channel's charge density
        is in CASTEP will be stored in the charge attribute while the 'down' or
        second spin channel's density will be stored in the spin attribute.
        Alternatively, the actual arrays can be returned by passing in ret_arrs=True.

        If the respective arrays are already there, this function simply returns them.

        Parameters
        ----------
        ret_arrs : bool
            return actual arrays for each spin channel rather than
            overriding the original ones in the class instance.

        Returns
        -------
        rhoup, rhodown : npt.NDArray[np.float64]
            charge arrays for spin up/down channel (if ret_arrs is True)

        Raises
        ------
        TypeError
            Data is not a density and thus has no 'rhoup' and 'rhodown'
        AssertionError
            Only one spin channel, hence no rhoup or rhodown.
        """

        if self.nspins == 1:
            raise AssertionError('Cannot get rhoup and rhodown as only 1 spin channel.')
        if self.is_den is False:
            raise TypeError('Cannot get rhoup and rhodown for non density data.')

        # Retrieve/calculate rhoup and rhodown appropriately.
        if self.have_rho_up_down is False:
            rhoup, rhodown = den_spin_to_rho_up_down(self.charge, self.spin)
        else:
            rhoup, rhodown = self.charge, self.spin

        if ret_arrs is True:
            return rhoup, rhodown
        elif self.have_rho_up_down is False:
            # Overwrite charge and spin arrays with rhoup and rhodown respectively.
            self.charge = rhoup
            self.spin = rhodown
            self.have_rho_up_down = True

    def get_charge_spin(self, ret_arrs: bool = False) -> Optional[Tuple[npt.NDArray[np.float64],
                                                                        npt.NDArray[np.float64]]]:
        """Get the total charge and spin density from spin channel densities.

        This assumes that rhoup is in the charge attribute and rhodown is the spin attribute.
        By default, this function will override the original arrays in the instance but the
        arrays can instead be returned without overriding by passing in ret_arrs=True.

        If the respective arrays are already there, this function simply returns them.

        Parameters
        ----------
        ret_arrs : bool
            return actual arrays for each spin channel rather than
            overriding the original ones in the class instance.

        Returns
        -------
        charge, spin : npt.NDArray[np.float64]
            total charge density and spin density arrays (if ret_arrs is True)

        Raises
        ------
        TypeError
            Data is not a density and thus has no 'charge' and 'spin' densities.
        AssertionError
            Only one spin channel, hence no spin density.
        """
        if self.nspins == 1:
            raise AssertionError('Cannot get charge and spin as only 1 spin channel.')
        if self.is_den is False:
            raise TypeError('Cannot get charge and spin for non density data.')

        # Retrieve/calculate charge and spin appropriately.
        # NB: charge array should be rhoup and spin array should be rhodown.
        if self.have_rho_up_down is True:
            charge, spin = rho_up_down_to_den_spin(self.charge, self.spin)
        else:
            charge, spin = self.charge, self.spin

        if ret_arrs is True:
            return charge, spin
        elif self.have_rho_up_down is True:
            # Overwrite charge and spin arrays.
            self.charge = charge
            self.spin = spin
            self.have_rho_up_down = False

    def shift_grid_data(self, frac_shift: npt.NDArray[np.float64]):
        """Perform cyclic shift on grid data with shift in frac. coords.

        The shift is provided in fractional coordinates. All allocated data arrays
        (charge, spin etc.) will be modified inplace.

        Parameters
        ----------
        frac_shift : npt.NDArray[np.float64]
            shift in fractional coordinates

        Raises
        ------
        IndexError
            frac_shift must be a 1D array of size 3.

        """
        return _frac_shift_grid(self, frac_shift)


def _frac_shift_grid(griddata: GridData, frac_shift: npt.NDArray[np.float64]) -> GridData:
    """Perform cyclic shift on grid data with shift in frac. coords.

    This is useful if you have used shifted cells in the calculation for convenience
    but make it hard to visualise the data.
    Also useful for magnetic systems, .e.g. antiferromagnets, with degenerate ground states.

    Parameters
    ----------
    griddata : GridData
        grid data to shift
    frac_shift : npt.NDArray[np.float64]
        shift in fractional coordinates

    Returns
    -------
    GridData
        shifted data

    Raises
    ------
    IndexError
        frac_shift must be a 1D array of size 3.

    """

    if frac_shift.ndim != 1 and frac_shift.shape[0] != 3:
        raise IndexError('frac_shift must be 1D array of size 3')

    # Calculate shift in grid coordinates.
    grid_shift = frac_shift * griddata.fine_grid

    def _do_shift(datarr):
        assert datarr.ndim == 3
        for i in range(3):
            datarr = np.roll(datarr, grid_shift[i], axis=i)
        return datarr

    # Shift grids as necessary
    # Densities
    if griddata.charge is not None:
        griddata.charge = _do_shift(griddata.charge)
    if griddata.spin is not None:
        griddata.spin = _do_shift(griddata.spin)
    if griddata.ncspin is not None:
        for ns in range(3):
            griddata.ncspin[ns, :, :, :] = _do_shift(griddata.ncspin[ns, :, :, :])

    # Potentials
    if griddata.pot is not None:
        for ns in range(griddata.nspins):
            griddata.pot[ns, :, :, :] = _do_shift(griddata.pot[ns, :, :, :])
    if griddata.ncpot is not None:
        for ns in range(3):
            griddata.ncpot[ns, :, :, :] = _do_shift(griddata.ncpot[ns, :, :, :])

    # Current plotting data - should always be allocated!
    griddata.cur_data = _do_shift(griddata.cur_data)

    return griddata
