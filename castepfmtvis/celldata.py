"""
This module handles the initialisation of unit cells.

All data is contained within a UnitCell class which can be initialised from:
1. reading the data from a .cell file used by CASTEP (note CASTEP arithmetic IS supported),
2. specifying the various required class attributes when initialising the class instance.

Author: Visagan Ravindran
"""
import warnings

import ase
import numpy as np
import numpy.typing as npt
import spglib

import castepfmtvis.arithmetic as arit
from castepfmtvis import io
from castepfmtvis.utils import cart_to_frac, frac_to_cart, reduce_frac_pts

__all__ = ['UnitCell', 'calc_recip_lat',
           'cell_cart_to_abc', 'cell_abc_to_cart']

_D2R = np.pi/180.0
_LENGTH_TOL = 5e-4  # angstroms
_ANGLE_TOL = 1e-2  # degrees


def _get_iprim(lengths, angles):
    """Get lattice vectors of body-centred cell.

    Body-centred cubic (BCC), orthorhombic body-centred (ORCI), tetragonal (BCT)
    have similar primitive cell vectors.
    """
    a, b, c = lengths
    alpha, beta, gamma = angles

    # Convert to radians
    alpha, beta, gamma = alpha*_D2R, beta*_D2R, gamma*_D2R

    x = a * np.cos(alpha/2)
    y = b * np.cos(beta/2)
    z = c * np.cos(gamma/2)

    real_lat = np.array([[-x, y, z],
                         [x, -y, z],
                         [x, y, -z]
                         ], dtype=np.float64)
    return real_lat


def _get_hex_cell(lengths, angles):
    """Construct hexagonal cell vectors given a set of lattice parameters."""
    a, b, c = lengths
    alpha, beta, gamma = angles

    # For hexagonal, a=b/=c (or some combination thereof) with alpha=beta=90
    # and gamma=60 or 120. We have to check which is the non-unique axis.
    if abs(a-b) < _LENGTH_TOL and \
       abs(alpha-90) < _ANGLE_TOL and abs(beta-90) < _ANGLE_TOL and\
       (abs(gamma-60) < _ANGLE_TOL or abs(gamma-120) < _ANGLE_TOL):
        uniq = 2
    elif abs(a-c) < _LENGTH_TOL and \
            abs(alpha-90) < _ANGLE_TOL and abs(gamma-90) < _ANGLE_TOL and\
            (abs(beta-60) < _ANGLE_TOL or abs(beta-120) < _ANGLE_TOL):
        uniq = 1
    elif abs(b-c) < _LENGTH_TOL and \
            abs(beta-90) < _ANGLE_TOL and abs(gamma-90) < _ANGLE_TOL and\
            (abs(alpha-60) < _ANGLE_TOL or abs(alpha-120) < _ANGLE_TOL):
        uniq = 0
    else:
        raise AssertionError(
            'Have HEX bravais lattice but could not find unique axis'
        )

    # Pick one lattice vector to align along the unique side
    # and then get axes for other two sides
    if uniq == 2:
        axis1, axis2 = 1, 0
    elif uniq == 1:
        axis1, axis2 = 2, 0
    elif uniq == 0:
        axis1, axis2 = 2, 1

    # Get the length of non-unique sides
    x = lengths[uniq - 1]

    # Now construct unit cell
    real_lat = np.zeros((3, 3), dtype=np.float64)

    # uniq=2, axis1=1, axis2=0
    real_lat[uniq, uniq] = lengths[uniq]
    real_lat[axis1, axis2] = 0.5*x
    real_lat[axis1, axis1] = np.sqrt(3.0)/2.0 * x
    if abs(angles[uniq] - 120.0 < _ANGLE_TOL):
        real_lat[axis2, axis2] = 0.5*x
        real_lat[axis2, axis1] = -np.sqrt(3.0)/2.0 * x
    elif abs(angles[uniq] - 60.0) < _ANGLE_TOL:
        real_lat[axis2, axis2] = x
    else:
        raise AssertionError('Inexact angle for hexagonal cell')

    return real_lat


def cell_abc_to_cart(lengths: npt.NDArray[np.float64],
                     angles: npt.NDArray[np.float64],
                     bv: str
                     ) -> npt.NDArray[np.float64]:

    a, b, c = lengths
    alpha, beta, gamma = angles

    real_lat = np.zeros((3, 3), dtype=np.float64)

    lengths_equal = abs(a-b) < _LENGTH_TOL and abs(a-c) < _LENGTH_TOL \
        and abs(b-c) < _LENGTH_TOL
    angles_equal = abs(alpha-beta) < _ANGLE_TOL and abs(alpha-gamma) < _ANGLE_TOL \
        and abs(beta-gamma) < _ANGLE_TOL

    if bv in ('CUB', 'BCC', 'FCC'):
        # Cubic lattices - check for primitive cells
        # FCC primitive if alpha=beta=gamma=60 and a=b=c
        prim_fcc = lengths_equal and angles_equal and abs(alpha-60) < _LENGTH_TOL
        # BCC primitive if alpha=beta=gamma with cos(alpha)=-1/3 and a=b=c
        prim_bcc = lengths_equal and angles_equal and \
            abs(alpha-np.arccos(-1/3)/_D2R) < _LENGTH_TOL

        if bv == 'FCC' and prim_fcc:
            real_lat[:, :] = a*np.sqrt(2)/2
            real_lat[0, 0], real_lat[1, 1], real_lat[2, 2] = 0.0, 0.0, 0.0
        elif bv == 'BCC' and prim_bcc:
            real_lat = _get_iprim(lengths, angles)
        else:
            # Eiter simple cubic or conventional cell, lattice vectors are trivial
            real_lat[0, 0], real_lat[1, 1], real_lat[2, 2] = a, b, c

    elif bv == 'HEX':
        real_lat = _get_hex_cell(lengths, angles)

    elif bv == 'RHL':
        warnings.warn('There can be an inconsistency between CASTEP setting and this setting, ' +
                      'specify your lattice vectors explicitly to ensure correct results.')
        k = 2.0*(1.0-np.cos(gamma * _D2R))
        x = a * np.sqrt(k)

        real_lat[0, 0] = x/np.sqrt(3.0)
        real_lat[1:, 0] = -0.5 * x/np.sqrt(3.0)
        real_lat[1, 1] = 0.5 * x
        real_lat[2, 1] = -0.5 * x
        real_lat[:, 2] = a*np.sqrt(1-k/3)

    elif bv == 'ORCI':
        # TODO For this we need to also check if it is a primitive cell which is tricky...
        raise NotImplementedError('ORCI not implemented for cell vectors from parameters')

    elif bv == 'ORCC':
        # TODO For this we need to also check if it is a primitive cell which is tricky...
        raise NotImplementedError('ORCC not implemented for cell vectors from parameters')

    elif bv == 'MCLC':
        # TODO For this we need to also check if it is a primitive cell which is tricky...
        raise NotImplementedError('MCLC not implemented for cell vectors from parameters')

    else:
        # Have a general Bravais lattice. We can generate a set of lattice vectors
        # (not the only unique set due to rotational DOF) by choosing
        # 1) a is along x
        # 2) b is along xy
        # 3) c is free (with the constraint that the square root is positive)
        # See K. N. Trueblood et al. Acta Cryst. (1996), A52, 770
        # and
        # McKie and McKie, "Essentials of Crystallography",
        # Blackwell Scientific Publishing, Oxford (1992)
        alpha_r, beta_r, gamma_r = alpha * _D2R, beta * _D2R, gamma * _D2R
        real_lat[0] = np.array([a, 0, 0], dtype=np.float64)
        real_lat[1] = b*np.array([np.cos(gamma_r), np.sin(gamma_r), 0])

        cx = c*np.cos(beta*_D2R)
        # this can be obtained from dot product with other two vectors...
        cy = c*(np.cos(alpha_r) - np.cos(beta_r) * np.cos(gamma_r))/np.sin(gamma_r)
        # Use Pythagoras, it's tidier!
        cz = np.sqrt(c**2 - cx**2 - cy**2)
        real_lat[2] = np.array([cx, cy, cz])

    return real_lat


def cell_cart_to_abc(real_lat: npt.NDArray[np.float64]) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Calculate lengths and angles in unit cell given the lattice vectors.

    Parameters
    ----------
    real_lat : np.ndarray
        lattice vectors (order is each vector is a 'row')

    Returns
    -------
    lengths : np.ndarray
        lengths of unit cell in angstroms
    angles : np.ndarray
        angles of unit cell in degrees
    """

    lengths = np.empty(3, dtype=np.float64)
    angles = np.empty(3, dtype=np.float64)

    # Calculate lengths -  rows in reallat are the vectors and columns are components
    for i, vec in enumerate(real_lat):
        lengths[i] = np.sqrt(np.sum(vec**2))

    # Calculate the cos of angles between each vector
    for i in range(3):
        # Angle i is related to dot product between i+1 and i+2 vectors (in cyclic order)
        vec1i, vec2i = (i+1) % 3, (i+2) % 3

        angles[i] = np.dot(real_lat[vec1i], real_lat[vec2i]) / \
            (lengths[vec1i] * lengths[vec2i])

    # Calculate angles (in degrees)
    angles = np.rad2deg(np.acos(angles))
    return lengths, angles


def _read_cell_pos(filename: str) -> tuple[list,
                                           npt.NDArray[np.float64] | None,
                                           npt.NDArray[np.float64] | None]:
    """Read atomic positions from CASTEP cell.

    Any supported arithmetic used to specify the positions is also parsed.
    This routine will return None for absolute/Cartesian positions cart_pos
    if we have fractional positions frac_pos and vice versa.

    Parameters
    ----------
    filename : str
        unit cell to read

    Returns
    -------
    species: list
        chemical species for each ion
    frac_pos : np.ndarray
        fractional positions for each ion
    cart_pos : np.ndarray
        cartesian/absolute positions in real-space for each ion
    """

    cart_pos, frac_pos = None, None
    have_frac = False
    # Try to find the POSITIONS_FRAC block and failing that POSITIONS_ABS block
    try:
        block_contents = io.read_block(filename, 'POSITIONS_FRAC')
    except IOError:
        block_contents = io.read_block(filename, 'POSITIONS_ABS')
    else:
        have_frac = True

    # Parse arithmetic in the block
    natoms = len(block_contents)
    species = [None for i in range(natoms)]

    vals = np.empty((natoms, 3), dtype=np.float64)
    for i, line in enumerate(block_contents):
        species[i] = line.split()[0]
        coords = line.split()[1:4]
        for j, val in enumerate(coords):
            vals[i, j] = arit.parse_arithmetic(val)

    # Store in appropriate array
    if have_frac is True:
        frac_pos = vals
    else:
        cart_pos = vals

    return species, frac_pos, cart_pos


def _get_bv_spg(cell: ase.Atoms) -> str:
    """Determine the high-symmetry points using the spacegroup.

    Note we do not use ASE's Bravais lattice as it is based solely on lattice parameters
    rather than the space group.  This is be particularly important in magnetic materials
    where the computational cell may have a different crystal system
    to the (conventional) crystallographic one.

    For low-symmetry Bravais lattices where the special/high-symmetry points are
    lattice parameter dependent, we will use the lattice parameters of the computational cell.

    Parameters
    ----------
    cell : ase.Atoms
        computational cell

    Returns
    -------
    bv_type: str
        Bravais lattice

    Raises
    ------
    IndexError
        Unknown Bravais lattice

    """

    # Get the space group information for this cell
    spg_cell = (cell.cell[:], cell.get_scaled_positions(), cell.get_atomic_numbers())
    spg_symb, spgno_str = spglib.get_spacegroup(spg_cell).split()

    # Remove the brackets returned around number in the above
    spg_no = int(spgno_str[spgno_str.find('(') + 1: spgno_str.find(')')])

    # Get the first letter of the spacegroup in international notation.
    bv_symb = spg_symb[0]

    # Now use the space group to determine the crystal system.
    # We can determine the actual Bravais lattice using the first
    # letter of the international notation symbol.
    #
    # Like in CASTEP, particularly for low symmetry Bravais lattices
    # where the high symmetry points depend on lattice parameters,
    # we will use the computational cell's lattice parameters.
    if 1 <= spg_no <= 2:
        # Triclinic lattice
        bv_type = 'TRI'
    elif 3 <= spg_no <= 15:
        # Monoclinic
        if bv_symb == 'P':  # Primitive monoclinic
            bv_type = 'MCL'
        elif bv_symb == 'C':  # Base-centred (C-centred) monoclinic
            bv_type = 'MCLC'
        else:
            raise IndexError(f'Unknown monoclinic lattice with space group: {bv_symb}')
    elif 16 <= spg_no <= 74:
        # Orthorhombic
        if bv_symb == 'P':  # Primitive Orthorhombic
            bv_type = 'ORC'
        elif bv_symb == 'I':  # Body-Centred Orthorhombic
            bv_type = 'ORCI'
        elif bv_symb == 'F':  # Face-Centred Orthorhombic
            bv_type = 'ORCF'
        elif bv_symb == 'A' or bv_symb == 'C':  # A/C-centred Orthorhombic
            bv_type = 'ORCC'
        else:
            raise IndexError(f'Unknown orthorhombic lattice with space group: {bv_symb}')
    elif 75 <= spg_no <= 142:
        # Tetragonal
        if bv_symb == 'P':  # Primitive Tetragonal
            bv_type = 'TET'
        elif bv_symb == 'I':  # Body-Centred Tetragonal
            bv_type = 'BCT'
        else:
            raise IndexError(f'Unknown tetragonal lattice with space group: {bv_symb}')
    elif 143 <= spg_no <= 167:
        # Trigonal
        if bv_symb == 'R':  # R-trigonal/Rhombohedral
            bv_type = 'RHL'
        elif bv_symb == 'P':  # Hexagonal
            bv_type = 'HEX'
        else:
            raise IndexError(f'Unknown trigonal lattice with space group: {bv_symb}')
    elif 168 <= spg_no <= 194:
        # Hexagonal
        bv_type = 'HEX'
    elif 195 <= spg_no <= 230:
        # Cubic
        if bv_symb == 'P':  # Primitive/Simple Cubic
            bv_type = 'CUB'
        elif bv_symb == 'I':  # Body-Centred Cubic
            bv_type = 'BCC'
        elif bv_symb == 'F':  # Face-Centred Cubic
            bv_type = 'FCC'
        else:
            raise IndexError(f'Unknown cubic lattice with space group: {bv_symb}')
    else:
        raise IndexError(f'Unknown Spacegroup {spg_no}: {bv_symb}')

    # Now get the Bravais lattice
    return bv_type


def _read_real_lat_cell(filename: str) -> npt.NDArray[np.float64]:
    """Read lattice vector from a cell file.

    Either one or of the LATTICE_CART and LATTICE_ABC blocks must be specified for this to work.
    """
    try:
        block_contents = io.read_block(filename, 'LATTICE_CART')
    except IOError:
        # No LATTICE_CART block so try LATTICE_ABC
        block_contents = io.read_block(filename, 'LATTICE_ABC')
        have_cart = False
    else:
        have_cart = True

    real_lat: npt.NDArray[np.float64] = np.empty((3, 3), dtype=np.float64)

    if have_cart is True:
        # Loop around the lines containing each vector parsing any arithmetic that may be present
        # in a given component.
        for i, vec in enumerate(block_contents):
            for j, comp in enumerate(vec.split()):
                real_lat[i, j] = arit.parse_arithmetic(comp)
    else:
        # Try to construct lattice vectors from lengths and angles.
        # First, we need to construct an ASE cell object so we need lattice positions
        symbols, scaled_positions, positions = _read_cell_pos(filename)

        # Now parse arithmetic in LATTICE_ABC block
        lengths = np.array([arit.parse_arithmetic(x) for x in block_contents[0].split()],
                           dtype=np.float64)
        angles = np.array([arit.parse_arithmetic(x) for x in block_contents[1].split()],
                          dtype=np.float64)

        # Create cell and then find the Bravais lattice we need
        cellparams = np.concatenate((lengths, angles))
        cell = ase.Atoms(
            symbols, positions=positions, scaled_positions=scaled_positions,
            cell=cellparams, pbc=True

        )
        bv = _get_bv_spg(cell)

        # Now that we have the Bravais lattice, construct lattice vectors.
        # NB: ASE and CASTEP use different conventions!
        real_lat = cell_abc_to_cart(lengths, angles, bv)

    return real_lat


def calc_recip_lat(real_lat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Calculate the reciprocal lattice vectors from the real lattice vectors.

    NB: The real lattice vectors in real_lat must be row-ordered.
    """
    recip_lat = np.empty((3, 3), dtype=np.float64)

    recip_lat[0] = np.cross(real_lat[1], real_lat[2])
    recip_lat[1] = np.cross(real_lat[2], real_lat[0])
    recip_lat[2] = np.cross(real_lat[0], real_lat[1])

    # Make sure to add pre-factor of 2 * Pi/V
    volume = np.dot(real_lat[0], recip_lat[0])  # a1 * (a2 x a3)
    recip_lat *= 2*np.pi/volume

    return recip_lat


class UnitCell():
    """Unit cell of a structure to visualise.

    Currently, the data can only be read or initialised in a CASTEP format.

    Attributes
    -------
    real_lat : np.ndarray
        The real row-ordered (i.e. real_lat[i,:] is the (i+1)th vector) lattice vectors
        (in Angstroms).
    recip_lat : np.ndarray
        The row-ordered reciprocal lattice vectors.
    species : list
        Chemical species for each ion.
    frac_pos : np.ndarray
        Atomic positions for each ion in fractional coordinates.
    cart_pos : np.ndarray
        Atomic positions for each ion in Cartesian coordinates (in Angstroms).

    Methods
    -------
    calc_cart_coord
        Convert a position given fractional coordinate to Cartesian coordinates.
    calc_frac_coord
        Convert a position given Cartesian coordinate to fractional coordinates.
    """

    def __init__(self, filename: str = None,
                 real_lat: npt.NDArray[np.float64] | None = None,
                 species: list | None = None,
                 frac_pos: npt.NDArray[np.float64] | None = None,
                 cart_pos: npt.NDArray[np.float64] | None = None,
                 reduce_frac: bool = True
                 ):
        """Initialise the unit cell.

        This can be read from a .cell file or set manually using the arguments.

        Parameters
        ----------
        filename : str
            CASTEP .cell file
        real_lat : npt.NDArray[np.float64] | None
            real lattice to use for the unit cell
        species : list | None
            chemical symbols for cell
        frac_pos : npt.NDArray[np.float64] | None
            fractonal positions for the atoms
        cart_pos : npt.NDArray[np.float64] | None
            Cartesian positions for the atoms

        Raises
        ------
        ValueError
            Missing necessary arguments
        """
        if species is None:
            if filename is None:
                raise ValueError('filename not provided')
            real_lat = _read_real_lat_cell(filename)

            # Now read the atomic positions
            species, frac_pos, cart_pos = _read_cell_pos(filename)
        else:
            if species is None:
                raise ValueError('species not provided')
            if real_lat is None:
                raise ValueError('real_lat not provided')
            if frac_pos is not None and cart_pos is not None:
                raise ValueError(
                    'Cannot provide both cart_pos and frac_pos at same time.'
                )
            if frac_pos is None and cart_pos is None:
                raise ValueError('Must provide one of cart_pos or frac_pos.')

        assert not (frac_pos is not None and cart_pos is not None)

        # Get reciprocal lattice
        self.real_lat = real_lat
        self.recip_lat = calc_recip_lat(self.real_lat)

        # Store atomic positions
        self.species = species
        self.nspecies = len(self.species)
        self.cart_pos = np.empty((self.nspecies, 3), dtype=np.float64)
        self.frac_pos = np.empty((self.nspecies, 3), dtype=np.float64)
        if cart_pos is None:
            # Have fractional coordinates so store them, then get Cartesian coordinates.
            assert frac_pos is not None
            # Rationalise fractional coordinates so that 0 <= frac_pos < 1 18/05/2025
            # Added option to disable rationalisation 09/06/2025
            if reduce_frac is True:
                frac_pos = reduce_frac_pts(frac_pos)

            self.frac_pos = frac_pos
            for i, pos in enumerate(frac_pos):
                self.cart_pos[i] = frac_to_cart(self.real_lat, pos)
        else:
            # Have Cartesian coordinates so store them, get fractional coordinates.
            self.cart_pos = cart_pos
            for i, pos in enumerate(cart_pos):
                self.frac_pos[i] = cart_to_frac(self.recip_lat, pos)

    def calc_cart_coord(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a position given fractional coordinate to Cartesian coordinates.

        Parameters
        ----------
        pos : npt.NDArray[np.float64]
            The position in fractional coordinates.

        Returns
        -------
        npt.NDArray[np.float64]
            The position in Cartesian coordinates.
        """
        return frac_to_cart(self.real_lat, pos)

    def calc_frac_coords(self, pos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Convert a position given Cartesian coordinate to fractional coordinates.

        Parameters
        ----------
        pos : npt.NDArray[np.float64]
            The position in Cartesian coordinates.

        Returns
        -------
        npt.NDArray[np.float64]
            The position in fractional coordinates.
        """
        return cart_to_frac(self.recip_lat, pos)
