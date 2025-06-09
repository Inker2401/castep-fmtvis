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

__all__ = ['GridData', 'read_castep_fmt', 'read_real_lat_fmt']


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

    units : str
        Units for the density or potential.
    charge: npt.NDArray[np.float64]
        Charge density (units: electrons*grid_pts)
    spin: npt.NDArray[np.float64]
        Spin density (units: spin*grid_pts)
    pot: npt.NDArray[np.float64]
        Potential for each (collinear) spin-channel (units: Hartrees)
    ncpot: npt.NDArray[np.float64]
        Potential for non-collinear spin-potential (units : Hartrees)
    """

    def __init__(self, filename: str | None = None,
                 is_den: bool | None = None,
                 nblank_header: int = 1,
                 real_lat: npt.NDArray[np.float64] = None,
                 dataarr: npt.NDArray[np.float64] = None,
                 units: str = ''
                 ):
        """
        Initialise rectilinear grid data.
        This can be initialised in two ways:
          1) Read formatted CASTEP rectilinear grid data
          2) Specifying the relevant attributes directly.
             NB: The grid data will be stored in the cur_data class.

        Parameters
        ----------
        filename : str
            formatted file to read
        is_den : bool | None
            Is file a density file?
            If not specified, it will be inferred from the file extension
        nblank_header : int
            number of blank lines after header

        """
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

            # Declare empty arrays first and allocate later
            self.charge: npt.NDArray[np.float64] | None = None
            self.spin: npt.NDArray[np.float64] | None = None
            self.ncspin: npt.NDArray[np.float64] | None = None
            self.pot: npt.NDArray[np.float64] | None = None
            self.ncpot: npt.NDArray[np.float64] | None = None
            self.cur_data: npt.NDArray[np.float64] | None = None

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
            if dataarr is None:
                raise ValueError('dataarr must be provided for direct initialisation')
            if dataarr.ndim != 3:
                raise IndexError('dataarr must be a 3D array')

            self.real_lat = real_lat
            self.cur_data = dataarr

            if len(units) == 0:
                self.units = 'a.u.'
            else:
                self.units = units.strip()

            # Set other necessary info
            self.fine_grid = self.cur_data.shape
            self.npts = np.prod(self.fine_grid)

            # Set other dummy info
            self.is_den = False
            self.nspins = 1
            self.have_nc = False

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
