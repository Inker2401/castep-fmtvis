"""
This file contains the global data defining the styles for ion spheres.
These are used when no user-specified styles are explicitly provided which use:
1. a colour scheme for each element's colour (currently support jmol, cpk and VESTA')
2. the radius is set by a van der Waals radius for the element.

Additionally, there is some code dealing with periodic boundary conditions.

Author: Visagan Ravindran
"""
import numpy as np
import numpy.typing as npt

# Define JMOL and CPK colour schemes for atomic spheres
# along with radii (using van der Waal radius with scale factor)
JMOL_COLOURS = {
    # period 1
    'H': '#FFFFFF', 'He': '#D9FFFF',
    # period 2
    'Li': '#CC80FF', 'Be': '#C2FF00', 'B': '#FFB5B5',
    'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'Ne': '#B3E3F5',
    # period 3
    'Na': '#AB5CF2', 'Mg': '#8AFF00', 'Al': '#BFA6A6',
    'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3',
    # period 4
    'K': '#8F40D4', 'Ca': '#3DFF00',
    # period 4 - transition metals
    'Sc': '#E6E6E6', 'Ti': '#BFC2C7', 'V': '#A6A6AB',
    'Cr': '#8A99C7', 'Mn': '#9C7AC7', 'Fe': '#E06633',
    'Co': '#F090A0', 'Ni': '#50D050', 'Cu': '#C88033',
    'Zn': '#7D80B0',
    # period 4 Remainder
    'Ga': '#C28F8F', 'Ge': '#668F8F', 'As': '#BD80E3',
    'Se': '#FFA100', 'Br': '#A62929', 'Kr': '#5CB8D1',
    # period 5
    'Rb': '#702EB0', 'Sr': '#00FF00',
    # period 5 - transition metals
    'Y': '#94FFFF', 'Zr': '#94E0E0', 'Nb': '#73C2C9',
    'Mo': '#54B5B5', 'Tc': '#3B9E9E', 'Ru': '#248F8F',
    'Rh': '#0A7D8C', 'Pd': '#006985', 'Ag': '#C0C0C0',
    'Cd': '#FFD98F',
    # period 5 - remainder
    'In': '#A67573', 'Sn': '#668080', 'Sb': '#9E63B5',
    'Te': '#D47A00', 'I': '#940094', 'Xe': '#429EB0',
    # period 6
    'Cs': '#57178F', 'Ba': '#00C900',
    # period 6 - lathanide series
    'La': '#70D4FF', 'Ce': '#FFFFC7', 'Pr': '#D9FFC7',
    'Nd': '#C7FFC7', 'Pm': '#A3FFC7', 'Sm': '#8FFFC7',
    'Eu': '#61FFC7', 'Gd': '#45FFC7', 'Tb': '#30FFC7',
    'Dy': '#1FFFC7', 'Ho': '#00FF9C', 'Er': '#00E675',
    'Tm': '#00D452', 'Yb': '#00BF38', 'Lu': '#00AB24',
    # period 6 - transition metals
    'Hf': '#4DC2FF', 'Ta': '#4DA6FF', 'W': '#2194D6',
    'Re': '#267DAB', 'Os': '#266696', 'Ir': '#175487',
    'Pt': '#D0D0E0', 'Au': '#FFD123', 'Hg': '#B8B8D0',
    # period 6 - remainder
    'Tl': '#A6544D', 'Pb': '#575961', 'Bi': '#9E4FB5',
    'Po': '#AB5C00', 'At': '#754F45', 'Rn': '#428296',
    # period 7
    'Fr': '#420066', 'Ra': '#007D00',
    # period 7 - actinide series
    'Ac': '#70ABFA', 'Th': '#00BAFF', 'Pa': '#00A1FF',
    'U': '#008FFF', 'Np': '#0080FF', 'Pu': '#006BFF',
    'Am': '#545CF2', 'Cm': '#785CE3', 'Bk': '#8A4FE3',
    'Cf': '#A136D4', 'Es': '#B31FD4', 'Fm': '#B31FBA',
    'Md': '#B30DA6', 'No': '#BD0D87', 'Lr': '#C70066',
    # period 7 - transition metals
    'Rf': '#CC0059', 'Db': '#D1004F', 'Sg': '#D90045',
    'Bh': '#E00038', 'Hs': '#E6002E', 'Mt': '#EB0026'
    # NOTE Missing jmol colours for elements 103 - 118
    # done at last ... unless they discover some new ones
}

CPK_COLOURS = {
    # period 1
    'H': '#FFFFFF', 'He': '#FFC0CB',
    # period 2
    'Li': '#B22222', 'Be': '#FF1493', 'B': '#00FF00',
    'C': '#C8C8C8', 'N': '#8F8FFF', 'O': '#F00000',
    'F': '#DAA520', 'Ne': '#FF1493',
    # period 3
    'Na': '#0000FF', 'Mg': '#228B22', 'Al': '#808090',
    'Si': '#DAA520', 'P': '#FFA500', 'S': '#FFC832',
    'Cl': '#00FF00', 'Ar': '#FF1493',
    # period 4
    'K': '#FF1493', 'Ca': '#808090',
    # period 4 - transition metals
    'Sc': '#FF1493', 'Ti': '#808090', 'V': '#FF1493',
    'Cr': '#808090', 'Mn': '#808090', 'Fe': '#FFA500',
    'Co': '#FF1493', 'Ni': '#A52A2A', 'Cu': '#A52A2A',
    'Zn': '#A52A2A',
    # period 4 - remainder
    'Ga': '#FF1493', 'Ge': '#FF1493', 'As': '#FF1493',
    'Se': '#FF1493', 'Br': '#A52A2A', 'Kr': '#FF1493',
    # period 5
    'Rb': '#FF1493', 'Sr': '#FF1493',
    # period 5 - transition metals
    'Y': '#FF1493', 'Zr': '#FF1493', 'Nb': '#FF1493',
    'Mo': '#FF1493', 'Tc': '#FF1493', 'Ru': '#FF1493',
    'Rh': '#FF1493', 'Pd': '#FF1493', 'Ag': '#808090',
    'Cd': '#FF1493',
    # period 5 - remainder
    'In': '#FF1493', 'Sn': '#FF1493', 'Sb': '#FF1493',
    'Te': '#FF1493', 'I': '#A020F0', 'Xe': '#FF1493',
    # period 6
    'Cs': '#FF1493', 'Ba': '#FFA500',
    # period 6 - lathanide series
    'La': '#FF1493', 'Ce': '#FF1493', 'Pr': '#FF1493',
    'Nd': '#FF1493', 'Pm': '#FF1493', 'Sm': '#FF1493',
    'Eu': '#FF1493', 'Gd': '#FF1493', 'Tb': '#FF1493',
    'Dy': '#FF1493', 'Ho': '#FF1493', 'Er': '#FF1493',
    'Tm': '#FF1493', 'Yb': '#FF1493', 'Lu': '#FF1493',
    # period 6 - transition metals
    'Hf': '#FF1493', 'Ta': '#FF1493', 'W': '#FF1493',
    'Re': '#FF1493', 'Os': '#FF1493', 'Ir': '#FF1493',
    'Pt': '#FF1493', 'Au': '#DAA520', 'Hg': '#FF1493',
    # period 6 - remainder
    'Tl': '#FF1493', 'Pb': '#FF1493', 'Bi': '#FF1493',
    'Po': '#FF1493', 'At': '#FF1493', 'Rn': '#FFFFFF',
    # period 7
    'Fr': '#FFFFFF', 'Ra': '#FFFFFF',
    # period 7 - actinide series
    'Ac': '#FFFFFF', 'Th': '#FF1493', 'Pa': '#FFFFFF',
    'U': '#FF1493', 'Np': '#FFFFFF', 'Pu': '#FFFFFF',
    'Am': '#FFFFFF', 'Cm': '#FFFFFF', 'Bk': '#FFFFFF',
    'Cf': '#FFFFFF', 'Es': '#FFFFFF', 'Fm': '#FFFFFF',
    'Md': '#FFFFFF', 'No': '#FFFFFF', 'Lr': '#FFFFFF'
    # NOTE Missing CPK colours for elements 103 - 118
    # done at last ... unless they discover some new ones
}

VESTA_COLOURS = {
    # period 1
    'H': '#FFCCCC', 'He': '#FCE8CE',
    # period 2
    'Li': '#86E074', 'Be': '#5ED77B', 'B': '#1FA20F',
    'C': '#804929', 'N': '#B0B9E6', 'O': '#FE0300',
    'F': '#B0B9E6', 'Ne': '#FE37B5',
    # period 3
    'Na': '#F9DC3C', 'Mg': '#FB7B15', 'Al': '#81B2D6',
    'Si': '#1B3BFA', 'P': '#C09CC2', 'S': '#FFFA00',
    'Cl': '#31FC02', 'Ar': '#CFFEC4',
    # period 4
    'K': '#A121F6', 'Ca': '#5A96BD',
    # period 4 - transition metals
    'Sc': '#B563AB', 'Ti': '#78CAFF', 'V': '#E51900',
    'Cr': '#00009E', 'Mn': '#A8089E', 'Fe': '#B57100',
    'Co': '#0000AF', 'Ni': '#B7BBBD', 'Cu': '#2247DC',
    'Zn': '#8F8F81',
    # period 4 - remainder
    'Ga': '#9EE373', 'Ge': '#7E6EA6', 'As': '#74D057',
    'Se': '#9AEF0F', 'Br': '#7E3102', 'Kr': '#FAC1F3',
    # period 5
    'Rb': '#FF0099', 'Sr': '#00FF26',
    # period 5 - transition metals
    'Y': '#66988E', 'Zr': '#00FF00', 'Nb': '#4CB276',
    'Mo': '#B386AF', 'Tc': '#CDAFCA', 'Ru': '#CFB7AD',
    'Rh': '#CDD1AB', 'Pd': '#C1C3B8', 'Ag': '#B7BBBD',
    'Cd': '#F21EDC',
    # period 5 - remainder
    'In': '#D780BB', 'Sn': '#9A8EB9', 'Sb': '#D7834F', 'Te': '#ADA251', 'I': '#8E1F8A', 'Xe': '#9AA1F8',
    # period 6
    'Cs': '#0EFEB9', 'Ba': '#1EEF2C',
    # period 6 - lathanide series
    'La': '#5AC449', 'Ce': '#D1FC06', 'Pr': '#FCE105',
    'Nd': '#FB8D06', 'Pm': '#0000F4', 'Sm': '#FC067D',
    'Eu': '#FA07D5', 'Gd': '#C003FF', 'Tb': '#7104FE',
    'Dy': '#3106FC', 'Ho': '#0741FB', 'Er': '#49723A',
    'Tm': '#0000E0', 'Yb': '#27FCF4', 'Lu': '#26FDB5',
    # period 6 - transition metals
    'Hf': '#B4B359', 'Ta': '#B79A56', 'W': '#8D8A7F',
    'Re': '#B3B08E', 'Os': '#C8B178', 'Ir': '#C9CE72',
    'Pt': '#CBC5BF', 'Au': '#FEB238', 'Hg': '#D3B7CB',
    # period 6 - remainder
    'Tl': '#95896C', 'Pb': '#52535B', 'Bi': '#D22FF7', 'Po': '#0000FF', 'At': '#0000FF', 'Rn': '#FFFF00',
    # period 7
    'Fr': '#000000', 'Ra': '#6DA958',
    # period 7 - actinide series
    'Ac': '#649E72', 'Th': '#25FD78', 'Pa': '#29FA35',
    'U': '#79A1AA', 'Np': '#4C4C4C', 'Pu': '#4C4C4C',
    'Am': '#4C4C4C', 'Cm': '#4C4C4C', 'Bk': '#4C4C4C',
    'Cf': '#4C4C4C', 'Es': '#4C4C4C', 'Fm': '#4C4C4C',
    'Md': '#4C4C4C', 'No': '#4C4C4C', 'Lr': '#4C4C4C',
    # period 7 - transition metals
    'Rf': '#4C4C4C', 'Db': '#4C4C4C', 'Sg': '#4C4C4C',
    'Bh': '#4C4C4C', 'Hs': '#4C4C4C', 'Mt': '#4C4C4C',
    'Ds': '#4C4C4C', 'Rg': '#4C4C4C', 'Cn': '#4C4C4C',
    # period 7 - remainder
    'Nh': '#4C4C4C', 'Fl': '#4C4C4C', 'Mc': '#4C4C4C',
    'Lv': '#4C4C4C', 'Ts': '#4C4C4C', 'Og': '#4C4C4C'
}

VDW_RADIUS = {
    # period 1
    'H': 1.10, 'He': 1.40,
    # period 2
    'Li': 1.82, 'Be': 1.53, 'B': 1.92,
    'C': 1.70, 'N': 1.55, 'O': 1.52,
    'F': 1.47, 'Ne': 1.54,
    # period 3
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84,
    'Si': 2.10, 'P': 1.80, 'S': 1.80,
    'Cl': 1.75, 'Ar': 1.88,
    # period 4
    'K': 2.75, 'Ca': 2.31,
    # period 4 - transition metals
    'Sc': 2.15, 'Ti': 2.11, 'V': 2.07,
    'Cr': 2.06, 'Mn': 2.05, 'Fe': 2.04,
    'Co': 2.00, 'Ni': 1.97, 'Cu': 1.96,
    'Zn': 2.01,
    # period 4 - remainder
    'Ga': 1.87, 'Ge': 2.11, 'As': 1.85,
    'Se': 1.90, 'Br': 1.85, 'Kr': 2.02,
    # period 5
    'Rb': 3.03, 'Sr': 2.49,
    # period 5 - transition metals
    'Y': 2.32, 'Zr': 2.23, 'Nb': 2.18,
    'Mo': 2.17, 'Tc': 2.16, 'Ru': 2.13,
    'Rh': 2.10, 'Pd': 2.10, 'Ag': 2.11,
    'Cd': 2.18,
    # period 5 - remainder
    'In': 1.93, 'Sn': 2.17, 'Sb': 2.06,
    'Te': 2.06, 'I': 1.98, 'Xe': 2.16,
    # period 6
    'Cs': 3.43, 'Ba': 2.68,
    # period 6 - lanthanide series
    'La': 2.43, 'Ce': 2.42, 'Pr': 2.40,
    'Nd': 2.39, 'Pm': 2.38, 'Sm': 2.36,
    'Eu': 2.35, 'Gd': 2.34, 'Tb': 2.33,
    'Dy': 2.31, 'Ho': 2.30, 'Er': 2.29,
    'Tm': 2.27, 'Yb': 2.26, 'Lu': 2.24,
    # period 6 - transition metals
    'Hf': 2.23, 'Ta': 2.22, 'W': 2.18,
    'Re': 2.16, 'Os': 2.16, 'Ir': 2.13,
    'Pt': 2.13, 'Au': 2.14, 'Hg': 2.23,
    # period 6 - remainder
    'Tl': 1.96, 'Pb': 2.02, 'Bi': 2.07,
    'Po': 1.97, 'At': 2.02, 'Rn': 2.20,
    # period 7
    'Fr': 3.48, 'Ra': 2.83,
    # period 7 - actinide series
    'Ac': 2.47, 'Th': 2.45, 'Pa': 2.43,
    'U': 2.41, 'Np': 2.39, 'Pu': 2.43,
    'Am': 2.44, 'Cm': 2.45, 'Bk': 2.44,
    'Cf': 2.45, 'Es': 2.45, 'Fm': 2.45,
    'Md': 2.46, 'No': 2.46, 'Lr': 2.46,
    # period 7 - transition metals
    # KLUDGE VdW radius is not known for these elements,
    # set to 2.50 angstroms
    'Rf': 2.50, 'Db': 2.50, 'Sg': 2.50,
    'Bh': 2.50, 'Hs': 2.50, 'Mt': 2.50,
    'Ds': 2.50, 'Rg': 2.50, 'Cn': 2.50,
    # period 7 - remainder
    'Nh': 2.50, 'Fl': 2.50, 'Mc': 2.50,
    'Lv': 2.50, 'Ts': 2.50, 'Og': 2.50
    # done at last ... unless they discover some new ones
}

################################
# PERIODIC BOUNDARY CONDITIONS
################################


def get_periodic_face(frac_pos: npt.NDArray[np.float64],
                      ion_no: int, element: str) -> tuple[
        npt.NDArray[np.float64] | None, str | None]:
    """Check if an atom lies on a face and impose periodic boundary conditions.

    An atom will lie on the face if its fractional coordinate is along each axis is
    a=0 or a=1,
    b=0 or b=1,
    c=0 or c=1,
    for a total of 6 possible combinations (faces).

    The input coordinates should be fractional coordinates!

    NB: The face_label returned is required for the sphere mesh to be created.

    Parameters
    ----------
    frac_pos : npt.NDArray[np.float64]
        The fractional coordinates of the ion.
    ion_no : int
        index for atoms
    element : str
        chemical symbol

    Returns
    ----------
    face_coord : npt.NDArray[np.float64]
       fractional coordinate of ion on the opposing face
       Returns None if ion not at face.
    face_label : str
       labels of for each periodic-imposed fractional coordinates (PyVista mesh labels)
       Returns None if ion not at face.
    """
    equal_tol = 1e-15  # tolerance check for equality
    face_coord, face_label = None, None
    a, b, c = frac_pos
    if abs(a - 0) < equal_tol:  # [0,b,c]
        # print('Copy face 1')
        face_coord = np.array([1, b, c])
        face_label = f'{element}_{ion_no}_face_1'

    elif abs(a - 1) < equal_tol:  # [1,b,c]
        # print('Copy face 2')
        face_coord = np.array([0, b, c])
        face_label = f'{element}_{ion_no}_face_2'

    elif abs(b - 0) < equal_tol:  # [a,0,c]
        # print('Copy face 3')
        face_coord = np.array([a, 1, c])
        face_label = f'{element}_{ion_no}_face_3'

    elif abs(b - 1) < equal_tol:  # [a,1,c]
        # print('Copy face 4')
        face_coord = np.array([a, 0, c])
        face_label = f'{element}_{ion_no}_face_4'

    elif abs(c - 0) < equal_tol:  # [a,b,0]
        # print('Copy face 5')
        face_coord = np.array([a, b, 1])
        face_label = f'{element}_{ion_no}_face_5'

    elif abs(c - 1) < equal_tol:  # [a,b,1]
        # print('Copy face 6')
        face_coord = np.array([a, b, 0])
        face_label = f'{element}_{ion_no}_face_6'

    return face_coord, face_label


def get_periodic_corner(frac_pos: npt.NDArray[np.float64],
                        ion_no: int, element: str) -> tuple[
        npt.NDArray[np.float64] | None, list | None]:
    """Check if an atom lies on a corner and impose periodic boundary conditions.

    There are 8 possible corners in the unit cell which we need to check (see below).
    NB: The input coordinates should be fractional coordinates!

    Parameters
    ----------
    frac_pos : npt.NDArray[np.float64]
        The fractional coordinates of the ion.
    ion_no : int
        index for atoms
    element : str
        chemical symbol

    Returns
    ----------
    corner_coords : npt.NDArray[np.float64]
       set of fractional coordinates for the atom at all other corners.
       Returns None if ion not at corner.
    corner_labels : list
       labels for ion mesh at each corner.
       Returns None if ion not at corner.
    """
    corner_coords, corner_labels = None, None
    corner_sharing = np.array([
        # 'Bottom'
        [0.0, 0.0, 0.0],  # lower left
        [1.0, 0.0, 0.0],  # lower right
        [0.0, 1.0, 0.0],  # upper left
        [1.0, 1.0, 0.0],  # upper right
        # 'Top'
        [0.0, 0.0, 1.0],  # lower left
        [1.0, 0.0, 1.0],  # lower right
        [0.0, 1.0, 1.0],  # upper left
        [1.0, 1.0, 1.0],  # upper right
    ], dtype=np.float64)

    # Determine which corner we have and then obtain the remainder.
    for i in range(8):  # corners
        if np.allclose(frac_pos, corner_sharing[i], rtol=1e-6, atol=1e-9):
            # Found explicitly provided corner, get the remaining corners
            # and exit loop
            corner_coords = np.delete(corner_sharing, i, 0)

            # Add labels for corner sharing
            corner_labels = [f'{element}_{ion_no}_corner_{j}' for j in range(7)]
            break

    return corner_coords, corner_labels
