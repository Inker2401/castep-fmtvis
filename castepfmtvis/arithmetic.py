"""
Handles parsing of arithmetic expressions in strings as done in CASTEP.

Author: Visagan Ravindran
"""
import re

import numpy as np

__all__ = ['parse_arithmetic', 'find_binary_op', 'find_funcs']

# Enable debugging messages for parser
DEBUG_PARSER = False


def parse_arithmetic(string: str) -> float:
    """Parse a string and convert it to a float parsing any arithmetic operations.

    The operations supported are those supported by CASTEP.
    Currently, the implementation assumes there are no spaces as in CASTEP in the string.

    Supported operations are:
    * trigonometric functions (input is degrees)
    * square roots
    * multiplication and division
    * addition and subtraction

    Parameters
    ----------
    string : str
        input string to parse and convert to float

    Returns
    -------
    float
        parsed floating-point number

    Raises
    ------
    ArithmeticError
        Was unable to parse/convert the string into a float
    IOError
        String consists of more than one word.
    """
    # Note original string for debugging
    ogstr = string

    if DEBUG_PARSER:
        print(f'parse_arithmetic: Parsing string:{ogstr}')

    # Check for number of words - there should only be one!
    string = string.strip()
    if len(string.split()) != 1:
        raise IOError(f'Multiple words found while parsing arithmetic in:"{ogstr}"')

    # Simplify double signs in the string
    string = _simplify_signs(string)

    # Follow order of operations - evaluate all operations of a given priority first
    # before going to the next one.
    # First do any trig or sqrt functions we might have
    string, success = find_funcs(string)
    if success is False:
        raise ArithmeticError(f'Failed so parse sqrt/trig functions in:"{ogstr}"')

    # Do multiplication and division...
    string, success = find_binary_op(string, False)
    if success is False:
        raise ArithmeticError(f'Failed so parse multiplication and division in:"{ogstr}"')

    # ... followed by addition and subtraction
    string, success = find_binary_op(string, True)
    if success is False:
        raise ArithmeticError(f'Failed so parse addition and subtraction in:"{ogstr}"')

    if DEBUG_PARSER:
        print(f'parse_arithmetic: Finished parsing string:{ogstr}={string}\n')

    # All arithmetic parsed, try converting number to float
    try:
        val = float(string)
    except ValueError as exc:
        errmsg = f'Unable to parse number, original expression: "{ogstr}"'
        raise ArithmeticError(errmsg) from exc

    return val


# ======================
# Auxilliary functions
# ======================
def _simplify_signs(string: str) -> str:
    """Simplify redundant signs in the expression.

    Examples of this are '--' which mathematically is equal to a '+'. As a bonus, this routine also
    checks if there are any illegal operations mathematically such as '+*'.

    Parameters
    ----------
    string : str
        string to simplify

    Returns
    -------
    str
        simplified string

    Raises
    ------
    ArithmeticError
        invalid mathematical operation found i.e. '+*', '+/', '-*', '-/'

    """

    # Pattern for signs that can be simplified or to look for
    # This includes invalid operations that we then catch later
    pattern = r'\+\+|--|\+-|-\+|\+\*|-\*|\+/|-/'

    string = string.strip()
    # Keep looping until we found all matches
    while m := re.search(pattern, string):
        # If found a simplifyable set of operators then, find out its location
        ops = m.group()
        loc = string.index(ops)

        # Determine simplification to make or check invalid operations
        if ops in ('+*', '-*', '+/', '-/'):
            raise ArithmeticError(f'Invalid operation found:"{ops}"')
        elif ops in ('++', '--'):
            replace = '+'
        elif ops in ('+-', '-+'):
            replace = '-'

        # Replace double signs with single sign
        string = string[:loc] + replace + string[loc+2:]

    return string


def _find_number(string: str, direct: str) -> str:
    """Find the first or last number in a string.

    This allows the number to be an (signed) integer, entered as a decimal or in scientific notation

    Parameters
    ----------
    string : str
        string to search
    direct : str
        direction to search string (F-forward, B-backward)

    Returns
    -------
    num : str
        number found

    Raises
    ------
    ValueError
        invalid value for direct

    """
    if direct == 'F':
        # Searching forward is easy, just grab the first matching pattern.
        # Define regex pattern for picking out a number allowing for the possibility
        # of scientific notation.
        numregex = r'[+-]?\d+(\.\d+)?([Ee][+-]?\d+)?'
        if m := re.search(numregex, string):
            num = m.group()

    elif direct == 'B':
        # To search backwards, we will reverse the string and then reverse the regex
        # pattern to take into account the number is backwards.
        # Note not to include the +/- sign at end as this can be treated as an operator.
        numregex = r'(?:\d+[+-]?[Ee])?(\d+\.)?\d+'

        # Reverse string and get number in reverse
        if m := re.search(numregex, string[::-1]):
            num = m.group()
            # Reverse number back to get it the correct way around
            num = num[::-1]

        # HACK: If the first character in a string is a sign and it is the only number
        # then, make sure to include it as a leading sign which will be missed by regex above.
        # For instance, if we have '-10', we need to return it as '-10'
        # This will not matter if it sandwiched between two numbers as it can be regarded as
        # an operator.
        if string[0] in ('+', '-') and string[1:] == num:
            num = string[0] + num
    else:
        raise ValueError('direct must be either "F" or "B"')
    return num


def _format_number(num: float) -> str:
    """Format a number in a string as a decimal or in scientific notation.

    This function should be used as opposed to manual formatting
    to avoid loss of precision when parsing as we convert numbers to strings
    during the process until parsing is complete.
    """
    # Define tolerances to use scientific notation
    mintol = 1e-8
    maxtol = 1e8

    # Format as decimal or use scientific notation for very small/large numbers
    if mintol < abs(num) < maxtol:
        numstr = f'{num:.16f}'
    else:
        numstr = f'{num:.16e}'

    return numstr


def find_binary_op(string: str, do_add_subtract: bool) -> tuple[str, bool]:
    """Evaluate binary operations between numbers.

    This evaluates all operations of a given priority, i.e.
    i) multiplication and division or,
    ii) addition and subtraction.
    This is called recursively until all operations of the priority are parsed.

    Parameters
    ----------
    string : str
        expression to parse
    do_add_subtract : bool
        look for addition and subtraction instead of multiplication and division

    Returns
    -------
    string : str
        string : parsed expression
    success : bool
        did we parse without encountering any errors?

    Raises
    ------
    ArithmeticError
        Divide by zero found in expression
    TypeError
        Could not parse number

    """

    div_zero_tol = 1e-100  # tolerance for divide by zero

    def _locate_operator(string: str) -> tuple[int, str]:
        # Set pattern for left-most operation of a given priority
        if do_add_subtract is True:
            if DEBUG_PARSER:
                print('find_binary_op: Parsing sums and differences on input:', string)
            # For addition and subtraction, +/- signs can appear as part of scientific
            # notation or as a leading sign.
            # Hence find the first instance which does not satisfy this, i.e.
            # the first +/- between two numbers and not preceded by an E.
            op_pattern = r'(?<![Ee])(?<=\d)[+-](?=\d)'
        else:
            if DEBUG_PARSER:
                print('find_binary_op: Parsing products and fractions on input:', string)
            # With multiplication and division, we basically just need to find
            # first instance between two numbers.
            op_pattern = r'(?<=\d)[\*/](?=[+-]?\d)'

        # Blank operator string and zero position
        op = ''
        op_pos = 0
        # Check if we have an operator of a specific priority
        if m := re.search(op_pattern, string):
            op_pos = m.start()
            op = m.group()

        return op_pos, op

    # Set success flag
    success = False

    # Simplify any double signs before proceeding.
    # This is particularly important for addition and subtraction
    string = _simplify_signs(string)  # this also strips whitespace

    # Find a match for the left most operator of a given priority
    op_pos, op = _locate_operator(string)
    if op_pos == 0:
        # If no operators of given priority found, then we have parsed
        # all operations of that priority so return
        success = True
        return string, success

    # If we had a match, get the part of the string before and after the operator
    # Expected format is [left][num1][op][num2][right]
    left = string[:op_pos]
    right = string[op_pos+1:]
    # if DEBUG_PARSER:
    #     print(f'find_binary_op: Initial {left=} ')
    #     print(f'find_binary_op: Initial {right=} ')

    # Get the last number in string before operator in left
    # and the first number in string after operator in right.
    num1 = _find_number(left, 'B')
    num2 = _find_number(right, 'F')

    # Remove the numbers from left and right parts of string to operator.
    if len(left)-len(num1) < 0:
        # String will wrap around itself so just set string to 0
        left = ''
    else:
        pos = len(left)-len(num1)
        left = left[:pos]
    right = right[len(num2):]
    if DEBUG_PARSER:
        print(f'find_binary_op: Found {left=} ')
        print(f'find_binary_op: Found {right=} ')
    # print(f'{left=} {right=}')

    # Try reading the two numbers, abort if we fail
    try:
        x = float(num1)
    except ValueError as exc:
        raise TypeError(f'Invalid first number for "{num1}{op}{num2}"') from exc
    try:
        y = float(num2)
    except ValueError as exc:
        raise TypeError(f'Invalid second number for "{num1}{op}{num2}"') from exc

    # Now evaluate the operation
    if op == '*':
        x *= y
    elif op == '/':
        # Check for divide by zero
        if abs(y) < div_zero_tol:
            raise ArithmeticError('Divide by zero found for expression:' +
                                  f'"{num1}{op}{num2}"')
        x /= y
    elif op == '+':
        x += y
    elif op == '-':
        x -= y

    # Convert the number back to a string and store into the original expression
    res = _format_number(x)
    string = left + res + right
    string = string.strip()

    if DEBUG_PARSER:
        print(f'find_binary_ops: {num1=} {op=} {num2=} {res=}')
        if do_add_subtract is True:
            print('find_binary_ops: Finished parsing sums and differences, new expr=',
                  string)
        else:
            print('find_binary_ops: Finished parsing products and fractions, new expr=',
                  string)

    # Everything's gone without a hitch so set success flag
    success = True

    # Check if there are any operators of this priority left and
    # use the superpower of recursion to parse them
    op_pos, *_ = _locate_operator(string)
    if op_pos != 0:
        string, success = find_binary_op(string, do_add_subtract)

    return string, success


def find_funcs(string: str) -> tuple[str, bool]:
    """Evaluate all trigonometric and square root functions in string.

    Parameters
    ----------
    string : str
        expression to parse

    Returns
    -------
    string : str
        string : parsed expression
    success : bool
        did we parse without encountering any errors?

    Raises
    ------
    TypeError
        Argument is not a number
    IOError
        Brackets are not closed for function
    """
    def _locfunc(string: str) -> tuple[int, str]:
        func, loc = '', 0
        funcregex = r'sin\(|cos\(|tan\(|sqrt\('
        if m := re.search(funcregex, string):
            func = m.group()[:-1]  # do not include final bracket
            loc = m.start() + len(func)  # position of first bracket
        return loc, func

    # Set status flag
    success = False

    # Simplify any double signs before proceeding.
    string = _simplify_signs(string)  # this also strips whitespace

    # See if we have any functions we need to
    loc1, func = _locfunc(string)
    if loc1 == 0:
        # If no match, then have parsed all functions so we can return now
        success = True  # make sure to set success!
        return string, success

    # We now find the bit of the string inside the bracket
    # Expected format is [head]func(numarg)[tail]
    head = string[:loc1-len(func)]
    # if DEBUG_PARSER:
    #     print(f'find_mathfunc: Found {head=}')

    # Get second bracket
    tmpstr = string[loc1:]
    if tmpstr.index(')') == 0:
        raise IOError(f'Brackets not closed for function "{func}"')

    # Get the global position of second bracket in the string
    loc2 = tmpstr.index(')') + len(func) + len(head)

    # Get the argument and the bit after the function
    tail = string[loc2+1:]
    arg = string[loc1+1:loc2]
    # if DEBUG_PARSER:
    #     print(f'find_mathfunc: Found {tail=}')

    # Now parse any arithmetic that may be in the argument
    arg, stat = find_binary_op(arg, False)
    if stat is False:
        return string, success
    arg, stat = find_binary_op(arg, True)
    if stat is False:
        return string, success

    # Finally evaluate the function
    try:
        x = float(arg)
    except ValueError as exc:
        raise TypeError(f'Failed to parse argument for {func}') from exc

    if func == 'sqrt':
        # Check if argument is negative before taking square root
        x = np.sqrt(x)
    elif func == 'sin':
        x = np.sin(np.deg2rad(x))
    elif func == 'cos':
        x = np.cos(np.deg2rad(x))
    elif func == 'tan':
        x = np.tan(np.deg2rad(x))

    # Convert number back into string and store in original expression
    res = _format_number(x)
    string = head + res + tail
    string = string.strip()

    if DEBUG_PARSER:
        print(f'find_mathfunc: Evaluated {func=} {arg=} {res=}')
        print('find_mathfunc: New expression:', string)

    # Everything's gone without a hitch so set success flag
    success = True

    # Check if we have any more functions to evaluate and if we do,
    # use the superpower of recursion
    loc1, *_ = _locfunc(string)
    if loc1 != 0:
        string, success = find_funcs(string)

    return string, success
