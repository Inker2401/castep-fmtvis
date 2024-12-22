"""
Test program for arithmetic parser
Author: V Ravindran
"""

import castepfmtvis.arithmetic as arit
import numpy as np

VERBOSE = True


def check_arithmetic(expr: str, expect: float) -> np.bool:
    """Check expression is evaluated correctly."""
    expr = expr.strip()
    val = arit.parse_arithmetic(expr)

    passed = np.isclose(val, expect)
    if VERBOSE is True:
        if passed:
            print(f'SUCCESS for:{expr}')
        else:
            print(f'FAILURE input:{expr}  evaluated:{val}  expected:{expect}')
    return passed


ngpass, ngtest = 0, 0  # passed/total test counters across all categories


def category_summary(category: str, npass: int, ntest: int):
    print(f'{category}: Passed {npass:n} out of {ntest:n} tests' +
          ' '*15 + '<-- SUMMARY')
    print(' ')


# =====================
# SINGLE NUMBER
# =====================
# Check if we can parse a sole number
category = 'SINGLE_NUMBERS'
exprs = ['+0.5', '-0.5', '1.0E2', '1.0E+2', '1E2', '-1E2']
expvals = [0.5, -0.5, 100.0, 100.0, 100.0, -100.0]
assert (len(exprs) == len(expvals))

npass, ntest = 0, len(exprs)
for expr, expect in zip(exprs, expvals):
    if check_arithmetic(expr, expect):
        npass += 1
category_summary(category, npass, ntest)

# Increment global test counter
ngpass += npass
ngtest += ntest

# ===========================
# MULTIPLICATION_AND_DIVISION
# ============================
# Check if we can do multiplication and division
category = 'MULTIPLICATION_AND_DIVISION'
exprs = ['20*5', '4*-5', '-5e0*2e+1', '200*+5*0.5*-0.25', '-4.0E-003/+5.0E+003',
         '20/5', '20/-4', '5/20', '5e-3/10e-3/1e+2'
         ]
expvals = [100.0, -20.0, -100.0, -125.0, -8e-7,
           4.0, -5.0, 0.25, 5e-3]
assert (len(exprs) == len(expvals))

npass, ntest = 0, len(exprs)
for expr, expect in zip(exprs, expvals):
    if check_arithmetic(expr, expect):
        npass += 1
category_summary(category, npass, ntest)

# Increment global test counter
ngpass += npass
ngtest += ntest

# ===========================
# ADDITION_AND_SUBTRACTION
# ============================
# Check if we can do addition and subtraction
category = 'ADDITION_AND_SUBTRACTION'
exprs = ['4.5+-5.0', '4.5++5.0', '4.5e-0--5.0e+0',
         '-4.5+3.25-2.75', '+4.5-+3.5-+2.5']
expvals = [-0.5, 9.5, 9.5,
           -4.0, -1.5]
assert (len(exprs) == len(expvals))

npass, ntest = 0, len(exprs)
for expr, expect in zip(exprs, expvals):
    if check_arithmetic(expr, expect):
        npass += 1
category_summary(category, npass, ntest)

# Increment global test counter
ngpass += npass
ngtest += ntest

# ============================
# MATHEMATICAL_FUNCTIONS
# ============================
category = 'FUNCTIONS'
exprs = [
    # Check if function definitions are correct first
    'sin(30)', 'cos(120)', 'tan(45)', 'sqrt(2)',
    # Check if we can parse arguments
    'sin(15*2)', 'cos(360/3)', 'tan(90/2)', 'sqrt(3*3*27/3)',
    # Check if we can do products and fractions of functions
    'sin(30)*cos(30)*sin(90)', 'sin(15*2)*cos(60/2)*sqrt(4*4)', 'tan(45)/sin(15*2)/cos(60/2)*sqrt(4*4)'
]

expvals = [0.5, -0.5, 1.0, np.sqrt(2),
           0.5, -0.5, 1.0, 9.0,
           np.sqrt(3.0)/4.0, np.sqrt(3.0), 16.0/np.sqrt(3.0)
           ]
assert (len(exprs) == len(expvals))

npass, ntest = 0, len(exprs)
for expr, expect in zip(exprs, expvals):
    if check_arithmetic(expr, expect):
        npass += 1
category_summary(category, npass, ntest)

# Increment global test counter
ngpass += npass
ngtest += ntest

# ============================
# MIXED_OPERATIONS
# ============================
category = 'MIXED_OPERATIONS'
exprs = [
    # Addition/subtraction mixed with multiplication and division
    '-2*5.25/10+4.5', '4.5-2*5.25/10',
    # Multiplication and division of trig functions and square roots
    '5*sqrt(2)*4.2e-2/cos(45)',
    '-2*sqrt(2)+sin(60)*cos(30)/sqrt(2)',
    # Extreme case!
    '-3.2+4e-3*cos(45)/tan(45)-sin(45)*sqrt(2)'
]

expvals = [3.45, 3.45,
           0.42,
           -2*np.sqrt(2)+np.sin(np.deg2rad(60))*np.cos(np.deg2rad(30))/np.sqrt(2),
           -3.2+4e-3*np.cos(np.deg2rad(45))/np.tan(np.deg2rad(45)) -
           np.sin(np.deg2rad(45))*np.sqrt(2)
           ]
assert (len(exprs) == len(expvals))

npass, ntest = 0, len(exprs)
for expr, expect in zip(exprs, expvals):
    if check_arithmetic(expr, expect):
        npass += 1
category_summary(category, npass, ntest)

# Increment global test counter
ngpass += npass
ngtest += ntest


print(f'Full test summary: Passed {ngpass:n} out of {ngtest:n} tests' +
      ' '*20 + '<-- SUMMARY')
