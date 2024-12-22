"""
Functions for reading formatted grid data from CASTEP files
as well as the various elements in CASTEP input files.
In addition, this file also handles the formatting of real numbers as fractions.

These functions serve as the building blocks of I/O routines within other modules.
Author: Visagan Ravindran
"""


def extract_header(filename: str,
                   start_head: str = 'BEGIN header',
                   end_head: str = 'END header',
                   case_insens: bool = False,
                   ret_lineno: bool = False
                   ) -> list | tuple[list, int, int]:
    """Extract the header from a CASTEP formatted file.

    The header should start with start_head and end with end_head.
    Both start_head and end_head are included in the header returned.
    By default, the search for the header is case-sensitive unless
    case_insens is passed in.

    Parameters
    ----------
    filename : str
        CASTEP formatted file to read
    start_head : str
        string that marks start of header
    end_head : str
        string that marks end of header
    case_insens : bool
        do case-insensitive search
    ret_lineno : bool
        return the line numbers in the file (starting from 0) that contain the header

    Returns
    -------
    header: list
        contents of header (including start_head and end_head)
    startline : int
        starting line number in file containing header
    endline : int
        ending line number in file containing header

    Raises
    ------
    IOError
        Could not find start or end of header in the file.

    """

    def __line_startwith(line: str, string: str, case_insens: bool):
        """Check if the line starts with a given string. """
        # Trim whitespaces and optionally turn to upper case for search.
        cur_line = line.strip()
        if case_insens is True:
            cur_line = cur_line.upper()
            string = string.upper()

        return cur_line.startswith(string)

    startline, endline = -1, -1
    header = []
    # Assume we have not found the various bits of the header yet.
    have_start, have_end = False, False

    with open(filename, 'r', encoding='ascii') as file:
        # The file could potentially be very big so rather than parsing in everything,
        # read line by line. The header should be near the top anyway!
        for n, line in enumerate(file):
            # Check if we the start of the header on the current line.
            if have_start is False:
                have_start = __line_startwith(line, start_head, case_insens)
                if have_start is True:
                    # Note the line no. of the starting line.
                    startline = n
                    header.append(line.strip())

            else:  # Have the start of the header so read it in.
                header.append(line.strip())

                # Check that we have not reached the end of the header
                have_end = __line_startwith(line, end_head, case_insens)
                if have_end is True:
                    # Note the line no. of the ending line and stop reading
                    endline = n
                    break

    if start_head is False:
        raise IOError(f'Reached EOF for {filename} but ' +
                      f'could not find start of header: "{start_head}"')
    if end_head is False:
        raise IOError(f'Reached EOF for {filename} but ' +
                      f'could not find end of header: "{end_head}"')

    if ret_lineno is True:
        return header, startline, endline
    else:
        return header


def read_block(filename: str, block_id: str) -> list:
    """Read the contents of a CASTEP block.

    The contents of the block should be delimitd by %BLOCK <block_id> and %ENDBLOCK <block_id>.
    Note per CASTEP standards, this function is case-insensitive in its search for the block.
    However, the actual block contents are returned in a case-insensitve manner.

    Parameters
    ----------
    filename : str
        file to read
    block_id : str
        CASTEP block data label

    Returns
    -------
    block_contents list
        contents of the block

    Raises
    ------
    IOError
        Could not find the specified block in the file

    """

    # Turn everything to upper case
    block_id = block_id.upper()

    # As this routine is intended primarily to be called on
    # short input files, we can just read it in its entirety.
    with open(filename, 'r', encoding='ascii') as file:
        lines = file.readlines()

    # Assume block is not found and then try to find it
    startline, endline = -1, -1
    for i, line in enumerate(lines):
        # Turn to upper case and split words
        split_line = line.upper().split()
        if split_line == ['%BLOCK', block_id]:
            startline = i
        if split_line == ['%ENDBLOCK', block_id]:
            endline = i

    if startline == -1:
        raise IOError(
            f'Unable to find %BLOCK {block_id.upper()} in file {filename}.'
        )
    if endline == -1:
        raise IOError(
            f'Unable to find %ENDBLOCK {block_id.upper()} in file {filename}.'
        )

    # Return without including  %BLOCK and %ENBDLOCK in file
    block_contents = lines[startline+1:endline]

    return block_contents
