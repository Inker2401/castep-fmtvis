"""
Check lattice vectors are generated correctly from lattice parameters
"""
import ase
import numpy as np
import spglib
from castepfmtvis import celldata

VERBOSE = False  # verbose output for debugging


def get_spacegroup(cell):
    spg_cell = (cell.cell[:], cell.get_scaled_positions(), cell.get_atomic_numbers())
    spg_symb, spgno_str = spglib.get_spacegroup(spg_cell).split()

    # Remove the brackets returned around number in the above
    spg_no = int(spgno_str[spgno_str.find('(') + 1: spgno_str.find(')')])

    return spg_no, spg_symb


def make_cell_vec(cellpar, symbols, scaled_positions):
    lengths = cellpar[:3]
    angles = cellpar[3:]

    # Create cell and then find the Bravais lattice we need
    cell = ase.Atoms(
        symbols, scaled_positions=scaled_positions,
        cell=cellpar, pbc=True
    )
    bv = celldata._get_bv_spg(cell)

    # Get space group
    spg_no, spg_symb = get_spacegroup(cell)

    if VERBOSE:
        print(f'Bravais lattice found: {bv}, spacegroup: {spg_no} {spg_symb}')

    # Now that we have the Bravais lattice, construct lattice vectors.
    # NB: ASE and CASTEP use different conventions!
    real_lat = celldata.cell_abc_to_cart(lengths, angles, bv)

    return real_lat, cell


def check_prim_conv_cell(convcell, primcell):
    """Checks that primitive cell is crystallographically equivalent to conventional cell"""
    # Check density of crystal
    convden = np.sum(convcell.get_masses())/convcell.get_volume()
    primden = np.sum(primcell.get_masses())/primcell.get_volume()
    if not np.isclose(convden, primden):
        if VERBOSE is True:
            print(f'Density of crystals do not match {convden=} {primden=}')
        return False

    # Check if spacegroup
    convspg, *_ = get_spacegroup(convcell)
    primspg, *_ = get_spacegroup(primcell)
    if convspg != primspg:
        if VERBOSE is True:
            print(f'Spacegroups do not match {convspg=} {primspg=}')
        return False

    # Both tests have passed so they are (probably...) crystallographically equivalent
    return True


npass, ntests = 0, 0

print('Checking cubic lattices')

# Simple cubic - BaTiO3
cellpar = np.array([4.0, 4.0, 4.0, 90.0, 90.0, 90.0])
symbols = ['Ba', 'Ti', 'O', 'O', 'O']
fracpos = np.array([
    [0, 0, 0], [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
])
calc_alat, *_ = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.diag([4.0, 4.0, 4.0])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: Simple cubic lattice')
    npass += 1
else:
    print('FAILURE: Simple cubic lattice')

# Body-centred cubic (BCC) - Cu (conventional cell)
cellpar = np.array([3.6150, 3.6150, 3.6150, 90.0, 90.0, 90.0])
symbols = ['Cu', 'Cu']
fracpos = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
calc_alat, convcell = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.diag([3.6150, 3.6150, 3.6150])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: BCC conventional lattice')
    npass += 1
else:
    print('FAILURE: BCC conventional lattice')

# Body-centred cubic (BCC) - Cu (primitive cell)
cellpar = np.array([3.6150*np.sqrt(3)/2, 3.6150*np.sqrt(3)/2, 3.6150*np.sqrt(3)/2,
                    109.47122063449069, 109.47122063449069, 109.47122063449069])
symbols = ['Cu']
fracpos = np.array([[0, 0, 0]])
calc_alat, primcell = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.array([[-3.6150/2, 3.6150/2, 3.6150/2],
                     [3.6150/2, -3.6150/2, 3.6150/2],
                     [3.6150/2, 3.6150/2, -3.6150/2]
                     ])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: BCC primitive lattice')
    npass += 1
else:
    print('FAILURE: BCC primitive lattice')

# Check if primitive cell and conventional cell match (i.e. is the test actually correct!)
ntests += 1
if check_prim_conv_cell(convcell, primcell):
    print('SUCCESS: BCC primitive and conventional cells match')
    npass += 1
else:
    print('FAILURE: BCC primitive and conventional cells do not match')

# Face-centred cubic (FCC) - Si (conventional cell)
cellpar = np.array([5.431, 5.431, 5.431, 90.0, 90.0, 90.0])
symbols = ['Si']*8
fracpos = np.array([[0, 0, 0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                    [0.25, 0.25, 0.25],
                    [0.75, 0.75, 0.25],
                    [0.75, 0.25, 0.75],
                    [0.25, 0.75, 0.75],
                    ])
calc_alat, convcell = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.diag([5.431, 5.431, 5.431])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: FCC conventional lattice')
    npass += 1
else:
    print('FAILURE: FCC conventional lattice')

# Face-centred cubic (FCC) - Si (primitive cell)
cellpar = np.array([5.431/np.sqrt(2), 5.431/np.sqrt(2),
                   5.431/np.sqrt(2), 60.0, 60.0, 60.0])
symbols = ['Si', 'Si']
fracpos = np.array([[0, 0, 0],
                    [0.25, 0.25, 0.25],
                    ])
calc_alat, primcell = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.array([
    [0.0, 5.431/2, 5.431/2],
    [5.431/2, 0.0, 5.431/2],
    [5.431/2, 5.431/2, 0.0]])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: FCC primitive lattice')
    npass += 1
else:
    print('FAILURE: FCC primitive lattice')

# Check if primitive cell and conventional cell match (i.e. is the test actually correct!)
ntests += 1
if check_prim_conv_cell(convcell, primcell):
    print('SUCCESS: FCC primitive and conventional cells match')
    npass += 1
else:
    print('FAILURE: FCC primitive and conventional cells do not match')

print('')

# Hexagonal (HEX) - BN
print('Checking hexagonal cells')
cellpar = np.array([2.512428, 2.512428, 7.707265,
                    90.0, 90.0, 120.0])
symbols = ['B', 'B', 'N', 'N']
fracpos = np.array([[1/3, 2/3, 1/4],
                    [2/3, 1/3, 3/4],
                    [1/3, 2/3, 3/4],
                    [2/3, 1/3, 1/4]
                    ])
calc_alat, cella = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.array([
    [1.2562140, -2.1758265, 0.0],
    [1.2562140, 2.1758265, 0.0],
    [0.0, 0.0, 7.707265]])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: HEX lattice (unique-axis c)')
    npass += 1
else:
    print('FAILURE: HEX lattice (unique-axis c)')

cellpar = np.array([2.512428, 7.707265, 2.512428,
                    90.0, 120.0, 90.0])
symbols = ['B', 'B', 'N', 'N']
fracpos = np.array([[1/3, 1/4, 2/3],
                    [2/3, 3/4, 1/3],
                    [1/3, 3/4, 2/3],
                    [2/3, 1/4, 1/3]
                    ])
calc_alat, cellb = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.array([
    [1.2562140, 0.0, -2.1758265,],
    [0.0, 7.707265, 0.0],
    [1.2562140, 0.0, 2.1758265]])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: HEX lattice (unique-axis b)')
    npass += 1
else:
    print('FAILURE: HEX lattice (unique-axis b)')

cellpar = np.array([7.707265, 2.512428, 2.512428,
                    120.0, 90.0, 90.0])
symbols = ['B', 'B', 'N', 'N']
fracpos = np.array([[1/4, 1/3, 2/3],
                    [3/4, 2/3, 1/3],
                    [3/4, 1/3, 2/3],
                    [1/4, 2/3, 1/3]
                    ])
calc_alat, cellc = make_cell_vec(cellpar, symbols, fracpos)
exp_alat = np.array([
    [7.7072650, 0.0, 0.0],
    [0.0, 1.2562140, -2.1758265],
    [0.0, 1.2562140, 2.1758265]])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: HEX lattice (unique-axis a)')
    npass += 1
else:
    print('FAILURE: HEX lattice (unique-axis a)')

# Check if all the hexagonal settings are equivalent
ntests += 1
if check_prim_conv_cell(cella, cellb) and check_prim_conv_cell(cella, cellc):
    print('SUCCESS: Hexagonal cells in different settings match')
    npass += 1
else:
    print('FAILURE: Hexagonal cells in different settings do not match')

print('')

# Check rhombohedral (conventional cell)
cellpar = np.array([5.415134, 5.415134, 13.428824,
                    90.0, 90.0, 120.0])
symbols = ['Bi', 'Bi', 'Bi', 'Bi', 'Bi', 'Bi',
           'Al', 'Al', 'Al', 'Al', 'Al', 'Al',
           'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O', 'O', 'O', 'O',
           'O', 'O', 'O', 'O']
fracpos = np.array([
    [0.0000000, 0.0000000, 0.0055850],
    [0.3333333, 0.6666667, 0.1722517],
    [0.6666667, 0.3333333, 0.3389183],
    [0.0000000, 0.0000000, 0.5055850],
    [0.3333333, 0.6666667, 0.6722517],
    [0.6666667, 0.3333333, 0.8389183],
    [0.0000000, 0.0000000, 0.2786100],
    [0.6666667, 0.3333333, 0.1119433],
    [0.6666667, 0.3333333, 0.6119433],
    [0.3333333, 0.6666667, 0.4452767],
    [0.3333333, 0.6666667, 0.9452767],
    [0.0000000, 0.0000000, 0.7786100],
    [0.3434393, 0.2213847, 0.2064357],
    [0.0101060, 0.4553880, 0.0397690],
    [0.7786153, 0.1220547, 0.2064357],
    [0.4452820, 0.9898940, 0.0397690],
    [0.5446120, 0.5547180, 0.0397690],
    [0.8779453, 0.6565607, 0.2064357],
    [0.0101060, 0.5547180, 0.5397690],
    [0.6767727, 0.7887213, 0.3731023],
    [0.4452820, 0.4553880, 0.5397690],
    [0.1119487, 0.3232273, 0.3731023],
    [0.2112787, 0.8880513, 0.3731023],
    [0.5446120, 0.9898940, 0.5397690],
    [0.6767727, 0.8880513, 0.8731023],
    [0.3434393, 0.1220547, 0.7064357],
    [0.1119487, 0.7887213, 0.8731023],
    [0.7786153, 0.6565607, 0.7064357],
    [0.8779453, 0.2213847, 0.7064357],
    [0.2112787, 0.3232273, 0.8731023]
])
calc_alat, convcell = make_cell_vec(cellpar, symbols, fracpos)
print(calc_alat)
exp_alat = np.array([
    [5.41513395,   0.0,   5.70577076E-008],
    [-2.70756698,   4.68964367,   5.70577076E-008],
    [-2.70756698,   -4.68964367,  5.70577076E-008]])
ntests += 1
if np.isclose(calc_alat, exp_alat).all():
    print('SUCCESS: RHL conventional cell')
    npass += 1
else:
    print('FAILURE: RHL conventional cell')
print(f'Passed a {npass} out of {ntests} tests')
