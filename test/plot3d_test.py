import castepfmtvis.plot as plt
import pyvista as pv
from castepfmtvis.celldata import UnitCell
from castepfmtvis.fmtdata import GridData

# Read the unit cell
cell = UnitCell('test.cell')

# Output some properties
print(' '*12 + 'Real Lattice (A)' + ' '*20+'Reciprocal Lattice(1/A)')
for a, b in zip(cell.real_lat, cell.recip_lat):
    print(f'{a[0]:12.6f}{a[1]:12.6f}{a[2]:12.6f}  {b[0]:14.8f}{b[1]:14.8f}{b[2]:14.8f}')

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

# Read density
den = GridData('test.den_fmt')
print('\nFine grid: ', den.fine_grid)
print('Have density: ', den.is_den)

###################
# Test static plot
###################
plotter = pv.Plotter()

# Make unit cell and add ions
plt.make_cell(plotter, cell)
plt.add_ions(plotter, cell)

# Add isosurface
isovalues = [0.00149, 0.00102, 0.000297]
names = [f'isosurface_den{i}' for i in range(len(isovalues))]
plt.plot_isosurface(plotter, den, names, isovalues, [
                    'blue', 'cyan', 'red'], opacity=[0.65, 0.4, 0.33],
                    labels=['covalent', '0.001 electrons', '0.0003 electrons']
                    )

# Add a legend
plotter.add_legend(bcolor='white', size=(0.15, 0.15),
                   border=True, face=None, name='atoms_legend')

# Display figure
plotter.show()

###################
# Test interactive
###################
plotter = pv.Plotter()

# Make unit cell and add ions
plt.make_cell(plotter, cell)
plt.add_ions(plotter, cell)

# Add two interactive isosurface
plt.interactive_isosurface(plotter, den, 'blue', 0.6, 'ch. den. (electrons)', 'iso1',
                           pointa=(0.1, 0.9), pointb=(0.6, 0.9))
plt.interactive_isosurface(plotter, den, 'red', 0.4, '', 'iso2',
                           pointa=(0.1, 0.8), pointb=(0.6, 0.8))
# Display figure
plotter.show()
