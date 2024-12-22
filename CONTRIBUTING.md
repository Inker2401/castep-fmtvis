# Contribution Guidelines for CASTEP Formatted Visualiser

## Coding Style
All code submitted should be in Python 3 (no Python 2 code please!).
More specifically, your code should be compatible with Python >=3.10 and PyVista>=0.40.0.

To ensure consistency, you should develop within a Python virtual environment
containing the necessary dependencies.

Instructions on how to create a virtual environment are outlined in the
[official Python documentation](https://docs.python.org/3.10/library/venv.html)

* In the instances where an error might occur, try to anticipate the error and raise the appropriate Python error class
  (with a meaningful message!).
  Try to provide as much information as possible as functions may be called within external Python scripts.
  Meaningful, detailed messages can aid in debugging.

* As far as possible, any functions should not depend on any global module variables and pass in all arguments to ensure that the code.

* That said, you should consider setting sensible default values for these arguments
  to minimise the number of arguments that would normally need to be specify
  (unless one wants a fine level of control over the plot).

* All functions should have a NumPy style docstring. For Emacs users, the package [numpydoc](https://github.com/douglasdavis/numpydoc.el) can assist with this.

* Functions should have type-hinting to indicate types as far as possible. Currently `mypy` is somewhat inconsistent with NumPy arrays but at a bare minimum, the docstring
  should indicate types (and array shapes if not obvious). Ideally, you should check arrays shapes, especially if other NumPy functions will not raise an error
  giving the false impression that everything is hunky-dory just because there are no errors!

* It is recommended that you run your code through a checker such as Pylint and Flake8.
  It is important to run these from the project's root directory as `setup.cfg` contains the necessary errors
  to ignore or raise and modify your code as necessary.

* Please format all code files using `autopep8`.  Likewise, run `autopep8' from the project root directory.


### Pull Request Procedure
At the bare minimum, your code should be error-free and not conflict with the master branch (i.e. fast-forwarding changes should be possible).
You can test you have not broken anything by ensuring all the Python files in the `test` folder can run without errors
along with all `examples`.
Since this is largely a visualisation based library, you should manually check the examples look sensible (before and after your changes!).

Fork the repository locally and make your changes on your fork.
Then pull from the master branch, rebase/merge as needed before pushing and submitting your merge request.

Once you have cloned your fork locally, change to the root project directory and then run the following command
`pip -e install .`
This will run the install command as usual but the `-e` or `editable` flag will allow you to make edits
and see your changes when you rerun the program rather than having to run the install command again.

## Examples
When adding new functionality, please provide an example in the `examples` folder.
Each example should be in its own folder.
This should consist of a:
- _well-documented_ Python script. The start of the script file should contain a header explaining what the example will teach or show.
- all necessary CASTEP input files.
  As far as possible, try to use existing cell files that are already present. This keeps the repository size small.
  When doing this, please symlink input files using **relative** paths from the example folder: `ln -s relative_path/to/file`
