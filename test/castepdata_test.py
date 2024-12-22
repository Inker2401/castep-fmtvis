"""
Check that the castepdata module works
Let's check that all attributes are set correctly as intended
on top of checking tha files are read correctly.

Author: Visagan Ravindran
"""
import numpy as np
from castepfmtvis.fmtdata import GridData

# Check if we can read charge densities
ch_den = GridData('test.den_fmt')
assert ch_den.is_den is True
assert ch_den.charge is not None and ch_den.nspins == 1 and ch_den.have_nc is False
assert np.isclose(ch_den.charge[0, 0, 0], 235.881121)
assert np.isclose(ch_den.charge[30, 26, 6], 51.133352)
assert np.isclose(ch_den.charge[9, 23, 16], 16.818431)
assert np.isclose(ch_den.charge[34, 35, 35], 27.202500)
assert np.isclose(ch_den.charge[35, 35, 35], 77.023717)
print('SUCCESSFULLY READ CHARGE DENSITY')

# Check if we can read local potentials
locpot = GridData('nospin.pot_fmt')
assert locpot.is_den is False
assert locpot.nspins == 1 and locpot.have_nc is False
assert locpot.pot is not None
assert np.isclose(locpot.pot[0, 20, 20, 20], -2.900180)
assert np.isclose(locpot.pot[0, 30, 20, 5], -0.086498)
assert np.isclose(locpot.pot[0, 9, 23, 6], -0.143155)
assert np.isclose(locpot.pot[0, 9, 23, 16], -0.490368)
assert np.isclose(locpot.pot[0, 34, 35, 35], 0.034852)
print('SUCCESSFULLY READ LOCAL POTENTIAL')

# Check if we can read can read charge and spin densities
spin_den = GridData('test_spin.den_fmt')
assert spin_den.is_den is True
assert spin_den.nspins == 2 and spin_den.have_nc is False
assert spin_den.charge is not None and spin_den.spin is not None
assert np.isclose(spin_den.charge[0, 0, 0], 0.037294) and \
    np.isclose(spin_den.spin[0, 0, 0], 0.037295)
assert np.isclose(spin_den.charge[30, 20, 5], 0.304386) and \
    np.isclose(spin_den.spin[30, 20, 5], 0.304386)
assert np.isclose(spin_den.charge[9, 23, 6], 0.296179) and \
    np.isclose(spin_den.spin[9, 23, 6], 0.296179)
assert np.isclose(spin_den.charge[9, 23, 16], 1.519513) and \
    np.isclose(spin_den.spin[9, 23, 16], 1.519513)
assert np.isclose(spin_den.charge[34, 35, 35], 0.072161) and \
    np.isclose(spin_den.spin[34, 35, 35], 0.072161)
assert np.isclose(spin_den.charge[39, 39, 38], 0.044832) and \
    np.isclose(spin_den.spin[39, 39, 38], 0.044833)
assert np.isclose(spin_den.charge[39, 39, 39], 0.042089) and \
    np.isclose(spin_den.spin[39, 39, 39], 0.042090)
print('SUCCESSFULLY READ CHARGE AND SPIN DENSITY')

# Check if we can read spin potentials
spin_pot = GridData('test_spin.pot_fmt')
assert spin_pot.is_den is False
assert spin_pot.nspins == 2 and spin_pot.have_nc is False
assert spin_pot.pot is not None
assert np.isclose(spin_pot.pot[0, 30, 20, 5], -0.107548) and \
    np.isclose(spin_pot.pot[1, 30, 20, 5], -0.064133)
assert np.isclose(spin_pot.pot[0, 9, 23, 6], -0.108213) and \
    np.isclose(spin_pot.pot[1, 9, 23, 6], -0.065665)
assert np.isclose(spin_pot.pot[0, 9, 23, 16], -0.219881) and \
    np.isclose(spin_pot.pot[1, 9, 23, 16], -0.125023)
assert np.isclose(spin_pot.pot[0, 34, 35, 35], -0.057691) and \
    np.isclose(spin_pot.pot[1, 34, 35, 35], -0.037401)
assert np.isclose(spin_pot.pot[0, 39, 39, 38], -0.045735) and \
    np.isclose(spin_pot.pot[1, 39, 39, 38], -0.029914)
print('SUCCESSFULLY READ SPIN POTENTIALS')
