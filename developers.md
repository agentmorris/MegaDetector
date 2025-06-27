# MegaDetector development conventions

## Contents

* [Coding conventions](#coding-conventions)
* [Linting](#linting)
* [Spell checking](#spell-checking)
* [Markdown link validation](#markdown-link-validation)
* [Tests](#tests)
* [Docs](#docs)


## Coding conventions

### Module header comment style

Python modules (.py files) should begin with a header comment, followed by a single blank line.

Example:

```python
"""

important_module.py

This module does important things.

"""

#%% This is the beginning of my code
```


### Closing comments

I use comments to indicate the end of large code blocks, including functions.  I know this is un-Pythonic.  I don't list all the arguments in a function in the closing comment, I just use (...) to represent the arguments.  Examples:

```python
def some_function(stuff):

    # This is a short loop, so I wouldn't really include a closing comment here,
    # but imagine it's very long.
    for thing in stuff:

        # This is a short loop, so I wouldn't really include a closing comment here,
        # but imagine it's very long.
        if isinstance(thing,str):
           print(thing)
        else:
           print(str(thing))
        # ...if [thing] is a string

    # ...for each thing

# ...def some_function(...)
```


### Cell delimiters

I use interactive execution during debugging, so I make extensive use of Python code cells, delimited by `#%%`.  For example:

```python
#%% Constants and imports

import sys
import os
```

The following cells are present in nearly every file:

* Constants and imports
* Classes
* Private functions
* Public functions
* Interactive driver
* Command-line driver

Cell delimiters should not be used within a class or function, since this prevents function definition by running a cell.  Logical blocks of code within a function should be defined with comments starting with `##`:

```python
def some_function(records):

    ## Validate arguments

    assert records is not None

    ## Explore the records

    # Find important records
    records_of_interest = []

    for i_record,record in enumerate(records):
        if (important):
            records_of_interest.append(record)

    # Print the important records
    for r in records_of_interest:
        print(str(r))
```


### Function and class header comment style (Google-style docstrings)

Functions and classes should have Google-style docstrings.  Docstrings should always be multi-line, even for very short functions or methods.  The single-line summary often required at the top of Google-style docstrings is not required.

Arguments should be specified as `name (type): description` or, for arguments with default values, `name (type, optional): description`.

Class attributes should be documented with inline comments, specifically using the `#: attribute_description` format, directly above or on the same line as the attribute initialization within the `__init__` method.

Example function:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    This is an example function.  The description does not need
    to start with a single blank line.

    Args:
        param1 (int): The first parameter, an integer.
        param2 (str): The second parameter, a string.

    Returns:
        A boolean value indicating success or failure.
    """

    # Function implementation
    return True
```

Example class:

```python
class ExampleClass:
    """
    This is an example class.

    It describes the overall purpose and behavior of the class.
    """

    def __init__(self, attr1: int, attr2: str):
        """
        Initializes ExampleClass.
        """

        #: The first attribute, an integer
        self.attr1 = attr1
        #: The second attribute, a string
        self.attr2 = attr2

    def example_method(self) -> None:
        """
        This is an example method.
        """

        # Method implementation
        pass
```


### Inline comment style

Inline comments should not end with periods unless they are full sentences.  Inline comments should almost never occur on the same line as code.

Examples:

```python
# This is a typical inline comment
a = b + c

# This is an inline comment that uses a full sentence.
x = y + z

p = q + r # Don't do this unless you absolutely have to
```


### Whitespace conventions

Consistent use of whitespace improves readability.

*   **Indentation:** Use four spaces for indentation.
*   **Blank lines:** Use blank lines to separate logical sections of code, such as functions, classes, and major blocks within functions. Typically, use two blank lines between top-level function and class definitions, and one blank line between methods in a class. A blank line should also always follow a function header comment (docstring) before the first line of code.
*   **Spaces around operators:** Use a single space on either side of binary operators (e.g., `+`, `-`, `*`, `/`, `=`, `==`, `!=`, `<`, `>`, `<=`, `>=`).
    *   Exception: No spaces around operators in keyword arguments or default parameter values (e.g., `func(param=value)`).
*   **Spaces after commas:** Use a single space after commas in argument lists, lists, tuples, and dictionaries.
*   **No spaces inside brackets/parentheses:** Avoid spaces immediately inside parentheses, brackets, or braces.
    *   Correct: `my_list = [1, 2, 3]`
    *   Incorrect: `my_list = [ 1, 2, 3 ]`
*   **No trailing whitespace:** Remove any trailing whitespace characters at the end of lines.
*   **Single newline at EOF:** Ensure all files end with a single newline character.

Example:

```python
def correct_spacing(param1, param2):
    """
    Example function for whitespace.
    """

    # Spaces around operator
    result = param1 + param2

    # Space after comma, no space inside brackets
    my_list = [1, 2, 3]

    return result


def another_function():
    """
    Another example function, with two blank lines after the previous function.
    """

    # One blank line before this block
    if x > 5:
        print('x is greater than 5')
```


### Line length

Aim for a maximum line length of 100-120 characters. While this is a guideline, prioritize readability. If breaking a line improves clarity, do so, even if it slightly exceeds the limit. Use backslashes or parens for line continuation for long statements.

Example:

```python
long_variable_name = \
    another_long_variable_name + \
    yet_another_variable - \
    some_other_value

if condition1 and condition2 and \
    condition3 or condition4):
    # code
    pass
```


### Parens

Prefer explicit parens in compounded conditions, e.g.:


```python
if (a is not None) and (b > 4):
    do_things()
```


### Naming conventions

*   **`snake_case`** for functions, methods, variables, and module names
    *   Example: `def calculate_area(radius):`, `image_width = 100`, `import data_utils`
*   **`CamelCase`** for class names
    *   Example: `class ImageProcessor:`
*   **`UPPER_SNAKE_CASE`** for constants
    *   Example: `MAX_ITERATIONS = 1000`, `DEFAULT_THRESHOLD = 0.5`


### Imports

Group imports in the following order:

1.  Standard library imports (e.g., `import os`, `import sys`)
2.  Third-party library imports (e.g., `import numpy as np`, `import tensorflow as tf`)
3.  Local application/library specific imports (e.g., `from . import utils`, `from megadetector.detection import run_detector`)

Within a group, lines starting with "from" should be placed after imports starting with "import".  Separate each group with a blank line.

Example:

```python
import os
import json
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from megadetector.utils import some_utility_function
```


### Quotes

Single quotes are preferred over double quotes when possible.


### String formatting

string.format() is preferred over f-strings when possible.


### Type hinting

Type hinting is encouraged for new code, but not required and not enforced retroactively.  Use standard Python type hints (PEP 484).

Example:

```python
def process_data(data: List[Dict[str, any]], threshold: float = 0.5) -> Optional[str]:
    """
    Processes a list of data dictionaries.

    Args:
        data (list): A list of dictionaries
        threshold (float): A float threshold for processing

    Returns:
        str: an optional string, or None if processing fails
    """

    if not data:
        return None

    return "processed"
```


## Linting

This repo uses [Ruff](https://docs.astral.sh/ruff/) to enforce coding conventions.

To install Ruff:

```bash
pip install ruff
```

To check the entire codebase for linting violations, navigate to the root of the repo and run:

```bash
ruff check .
```

To fix linting violations:

```bash
ruff check . --fix
```


## Spell checking

I use the [typos](https://pypi.org/project/typos/) package for spell checking:

```bash
mamba install typos -y
cd ~/git/MegaDetector
typos --config ~/git/MegaDetector/envs/typos.toml --exclude \*.resx --exclude \*.ipynb --exclude \*.js --exclude archive\* | more
```


## Markdown link validation

I use [markdown-link-validator](https://www.npmjs.com/package/markdown-link-validator) for link validation in .md files:

```bash
npm install -g markdown-link-validator
cd ~/git/MegaDetector
markdown-link-validator .
```

## Testing

### Test data

A zipfile of camera trap images is available at:

<https://lila.science/public/md-test-package.zip>

This contains:

* Images with unusual filenames
* Images that are corrupted
* Images with animals
* Images with people
* Videos


### Testing conventions

* Modules should have a cell called `#%% Tests`.  This should be the last cell in each module.  Functions within that cell should start with `test_`.
* Tests that require temporary folders should create them using `ct_utils.make_test_folder()`, which keeps temporary folders under a `megadetector/tests` folder within the system temp folder.  Temporary folders should be cleaned up after tests are complete.

### Running all tests

This repo uses `pytest` for testing; cd to the repo root and run:

`pytest -v`

...to run tests.

### Running the most important tests

The most important tests - the ones about actually running models - are in md_tests.py.  These are run by the automated test suite, but to run them manually, you can do something like:

```bash
set PYTHONPATH=c:\git\MegaDetector;c:\git\yolov5-md
mamba activate megadetector
python c:\git\MegaDetector\megadetector\utils\md_tests.py --cli_working_dir "c:\git\MegaDetector" --cli_test_pythonpath "c:\git\MegaDetector;c:\git\yolov5-md" --max_coord_error 0.01 --max_conf_error 0.01 --skip_cpu_tests --skip_video_tests --skip_download_tests
```

This also tests the CLI entry points, which pytest does not.

## Docs

This repo uses [Sphinx](https://www.sphinx-doc.org/en/master/) to build documentation.  To build the docs, cd to the repo root, then:

```bash
mamba env create -f envs/environment-docs.yml
mamba activate megadetector-docs
cd docs
make clean && make html
```
