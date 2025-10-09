"""
Check that the castepdata module works
Let's check that all attributes are set correctly as intended
on top of checking tha files are read correctly.

Author: Visagan Ravindran
"""
import numpy as np
from castepfmtvis.fmtdata import GridData
from castepfmtvis.celldata import UnitCell
from ase.units import Bohr

WRITE_DEBUG_CELL = False
WRITE_DEBUG_GRIDDATA = False


def dump_griddata(griddata: GridData):
    def allocated(arr):
        if arr is None:
            return 0
        else:
            return arr.shape
    print('Fine grid: ', griddata.fine_grid)
    print('Number of grid points: ', griddata.npts)
    print('Have density: ', griddata.is_den)
    print('No of spins: ', griddata.is_den)
    print('Have non-collinear: ', griddata.have_nc)
    print('Real lattice: ', griddata.real_lat)
    print('Shape (charge,spin,ncspin): ',
          allocated(griddata.charge), allocated(griddata.spin), griddata.ncspin)
    print('Shape (pot,ncpot): ',
          allocated(griddata.pot), allocated(griddata.ncpot))


def init_griddata(src: GridData) -> GridData:
    """Initialise a griddata instance manually based on source data."""
    outdata = GridData(real_lat=filedata.real_lat, datarr=filedata.cur_data,
                       is_den=filedata.is_den,
                       nspins=filedata.nspins, have_nc=filedata.have_nc,
                       charge=filedata.charge, spin=filedata.spin,
                       ncspin=filedata.ncspin,
                       pot=filedata.pot, ncpot=filedata.ncpot
                       )
    return outdata


def check_den(den: GridData):
    """Test density is read correctly."""
    if WRITE_DEBUG_GRIDDATA is True:
        dump_griddata(den)
    assert den.is_den is True
    assert den.have_rho_up_down is False
    assert den.charge is not None
    assert den.have_nc is False
    assert den.nspins == 1
    assert np.isclose(den.charge[0, 0, 0], 235.881121)
    assert np.isclose(den.charge[30, 26, 6], 51.133352)
    assert np.isclose(den.charge[9, 23, 16], 16.818431)
    assert np.isclose(den.charge[34, 35, 35], 27.202500)
    assert np.isclose(den.charge[35, 35, 35], 77.023717)


def check_pot(poten: GridData):
    """Test potential is read correctly."""
    if WRITE_DEBUG_GRIDDATA is True:
        dump_griddata(poten)
    assert poten.is_den is False
    assert poten.nspins == 1
    assert poten.have_nc is False
    assert poten.pot is not None
    assert np.isclose(poten.pot[0, 20, 20, 20], -2.900180)
    assert np.isclose(poten.pot[0, 30, 20, 5], -0.086498)
    assert np.isclose(poten.pot[0, 9, 23, 6], -0.143155)
    assert np.isclose(poten.pot[0, 9, 23, 16], -0.490368)
    assert np.isclose(poten.pot[0, 34, 35, 35], 0.034852)


def check_den_spin(den: GridData):
    """Test a spin-polarised density data is read correctly."""
    if WRITE_DEBUG_GRIDDATA is True:
        dump_griddata(den)
    assert den.is_den is True
    assert den.have_rho_up_down is False
    assert den.have_nc is False
    assert den.nspins == 2
    assert den.charge is not None
    assert den.spin is not None

    assert np.isclose(den.charge[0, 0, 0], 0.037294) and \
        np.isclose(den.spin[0, 0, 0], 0.037295)
    assert np.isclose(den.charge[30, 20, 5], 0.304386) and \
        np.isclose(den.spin[30, 20, 5], 0.304386)
    assert np.isclose(den.charge[9, 23, 6], 0.296179) and \
        np.isclose(den.spin[9, 23, 6], 0.296179)
    assert np.isclose(den.charge[9, 23, 16], 1.519513) and \
        np.isclose(den.spin[9, 23, 16], 1.519513)
    assert np.isclose(den.charge[34, 35, 35], 0.072161) and \
        np.isclose(den.spin[34, 35, 35], 0.072161)
    assert np.isclose(den.charge[39, 39, 38], 0.044832) and \
        np.isclose(den.spin[39, 39, 38], 0.044833)
    assert np.isclose(den.charge[39, 39, 39], 0.042089) and \
        np.isclose(den.spin[39, 39, 39], 0.042090)


def check_pot_spin(poten: GridData):
    """Test a spin-polarised potential data is read correctly."""
    if WRITE_DEBUG_GRIDDATA is True:
        dump_griddata(poten)
    assert poten.is_den is False
    assert poten.nspins == 2
    assert poten.have_nc is False
    assert poten.pot is not None
    assert np.isclose(poten.pot[0, 30, 20, 5], -0.107548) and \
        np.isclose(poten.pot[1, 30, 20, 5], -0.064133)
    assert np.isclose(poten.pot[0, 9, 23, 6], -0.108213) and \
        np.isclose(poten.pot[1, 9, 23, 6], -0.065665)
    assert np.isclose(poten.pot[0, 9, 23, 16], -0.219881) and \
        np.isclose(poten.pot[1, 9, 23, 16], -0.125023)
    assert np.isclose(poten.pot[0, 34, 35, 35], -0.057691) and \
        np.isclose(poten.pot[1, 34, 35, 35], -0.037401)
    assert np.isclose(poten.pot[0, 39, 39, 38], -0.045735) and \
        np.isclose(poten.pot[1, 39, 39, 38], -0.029914)


print('Testing ability to read formatted data files')
# Check if we can read/initialise charge densities
filedata = GridData('test.den_fmt')
check_den(filedata)
print('SUCCESSFULLY read charge density')
mygriddata = init_griddata(filedata)
check_den(mygriddata)
print('SUCCESSFULLY initialised charge density')

# Check if we can read local potentials
filedata = GridData('nospin.pot_fmt')
check_pot(filedata)
print('SUCCESSFULLY read local potential')
mygriddata = init_griddata(filedata)
check_pot(mygriddata)
print('SUCCESSFULLY initialised local potential')

# Check if we can read can read charge and spin densities
filedata = GridData('test_spin.den_fmt')
check_den_spin(filedata)
print('SUCCESSFULLY read charge and spin density')
mygriddata = init_griddata(filedata)
check_den_spin(mygriddata)
print('SUCCESSFULLY initialised charge and spin density')

# Check if we can read spin potentials
filedata = GridData('test_spin.pot_fmt')
check_pot_spin(filedata)
print('SUCCESSFULLY read spin potentials')
mygriddata = init_griddata(filedata)
check_pot_spin(mygriddata)
print('SUCCESSFULLY initialised spin potentials')


def write_cell_contents(cell: UnitCell):
    """Debug cell contents."""
    print(' '*12 + 'Real Lattice (A)' + ' '*20+'Reciprocal Lattice(1/A)')
    for v in cell.real_lat:
        u = v / Bohr
        print(f'{v[0]:12.6f}{v[1]:12.6f}{v[2]:12.6f} ' +
              f'{u[0]:12.6f}{u[1]:12.6f}{u[2]:12.6f}')
    print('\n'+' '*4+'Number of ions ', cell.nspecies)
    print('-'*80)
    print(' '*30+'Cell contents')
    print('-'*80)
    print(' '*11+'Fractional Coordinates'+' '*16+'Cartesian Coordinates(A)')
    for i in range(cell.nspecies):
        sp = cell.species[i]
        frac = cell.frac_pos[i]
        cart = cell.cart_pos[i]
        print(f'{sp:2} {frac[0]:12.8f}{frac[1]:12.8f}{frac[2]:12.8f}  ' +
              f'{cart[0]:12.8f}{cart[1]:12.8f}{cart[2]:12.8f}')


# Initialise reference cell - this should also catch if we can do direct initialisation
print('\nTesting ability to read cell files')
ref_cell = UnitCell(
    real_lat=np.array([
        [0.0, 2.715500, 2.715500],
        [2.715500, 0.0, 2.715500],
        [2.715500, 2.715500, 0.0]]),
    species=['Si', 'Si'],
    frac_pos=np.array([
        [0.0, 0.0, 0.0], [1/4, 1/4, 1/4]])
)


def check_cell(cellfile):
    """Check cell initialises correctly."""
    cell = UnitCell(cellfile)
    if WRITE_DEBUG_CELL is True:
        print('\nTesting cell: ', cellfile)
        write_cell_contents(cell)
        print('')

    assert np.isclose(ref_cell.real_lat, cell.real_lat).all()
    assert np.isclose(ref_cell.recip_lat, cell.recip_lat).all()
    assert ref_cell.nspecies == cell.nspecies
    assert ref_cell.species == cell.species
    assert np.isclose(ref_cell.frac_pos, cell.frac_pos, atol=1e-11, rtol=5e-7).all()
    assert np.isclose(ref_cell.cart_pos, cell.cart_pos).all()
    print(f'SUCCESSFULLY READ {cellfile}')


# Check that we can read cell files
# NB: Only LATTICE_CART and LATTICE_ABC blocks supports units for now, POSITIONS_ABS must be in Angstroms!
cellfiles = ('test_cells/lat_abc_ang.cell', 'test_cells/lat_abc_bohr.cell',
             'test_cells/lat_cart_ang.cell', 'test_cells/lat_cart_bohr.cell')
for f in cellfiles:
    check_cell(f)
