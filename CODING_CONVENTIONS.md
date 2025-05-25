# MegaDetector coding conventions

## Module header comment style

Python modules (.py files) should begin with a header comment:

Example:

```python
"""

important_module.py

This module does important things.

"""
```

## Function and class header comment style (Google-style docstrings)

Functions and classes should have Google-style docstrings.  Docstrings should always be multi-line, even for very short functions or methods.

Class attributes should be documented with inline comments, specifically using the `#: attribute_description` format, directly above or on the same line as the attribute initialization within the `__init__` method.

Example function:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    This is an example function.

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

## Inline comment style

Inline comments should not end with periods unless they are full sentences.  Inline comments should almost never occur on the same line as code.

Examples:

```python
# This is a typical inline comment
a = b + c

# This is an inline comment that uses a full sentence.
x = y + z

p = q + r # Don't do this unless you absolutely have to
```

## Whitespace conventions

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

## Line length

Aim for a maximum line length of 100-120 characters. While this is a guideline, prioritize readability. If breaking a line improves clarity, do so, even if it slightly exceeds the limit. Use parentheses for implied line continuation for long statements.

Example:

```python
long_variable_name = (
    another_long_variable_name +
    yet_another_variable -
    some_other_value
)

if (condition1 and condition2 and
        condition3 or condition4):
    # code
    pass
```

## Naming conventions

*   **`snake_case`:** For functions, methods, variables, and module names.
    *   Example: `def calculate_area(radius):`, `image_width = 100`, `import data_utils`
*   **`PascalCase` (or `CapWords`):** For class names.
    *   Example: `class ImageProcessor:`
*   **`UPPER_SNAKE_CASE`:** For constants.
    *   Example: `MAX_ITERATIONS = 1000`, `DEFAULT_THRESHOLD = 0.5`

## Imports

Group imports in the following order:

1.  Standard library imports (e.g., `import os`, `import sys`)
2.  Third-party library imports (e.g., `import numpy as np`, `import tensorflow as tf`)
3.  Local application/library specific imports (e.g., `from . import utils`, `from megadetector.detection import run_detector`)

Separate each group with a blank line.

Example:

```python
import os
import json
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from megadetector.utils import some_utility_function
```

## Type hinting

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

    return "Processed"
```


## Linting with Ruff

We use [Ruff](https://docs.astral.sh/ruff/) to enforce coding conventions.

### Installation

To install Ruff:

```bash
pip install ruff
```

### Checking for violations

To check the entire codebase for any linting violations, navigate to the root of the repository and run:

```bash
ruff check .
```

### Automatically fixing violations

Ruff can attempt to automatically fix many of the violations it finds. To do this recursively for the current folder:

```bash
ruff check . --fix
```
