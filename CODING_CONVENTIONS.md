# Coding conventions

## Introduction

This document outlines the coding conventions to be followed when contributing to the MegaDetector project. Adhering to these conventions ensures consistency, readability, and maintainability of the codebase.

## Module header comment style

All Python modules (.py files) should begin with a header comment that provides a brief overview of the module's purpose.

Example:

```python
#
# CODING_CONVENTIONS.md
#
# Outlines the coding conventions for the MegaDetector project.
#
```

## Function and class header comment style (Google-style docstrings)

All functions and classes should have Google-style docstrings that clearly explain their purpose, arguments, and return values (for functions) or methods (for classes). Docstrings should always be multi-line, even for very short functions or methods.

Class attributes should be documented with inline comments, specifically using the `#: attribute_description` format, directly above or on the same line as the attribute initialization within the `__init__` method.

Example (Function):

```python
def example_function(param1: int, param2: str) -> bool:
    """
    This is an example function.

    Args:
        param1: The first parameter, an integer.
        param2: The second parameter, a string.

    Returns:
        A boolean value indicating success or failure.
    """

    # Function implementation
    return True
```

Example (Class):

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

        #: The first attribute, an integer.
        self.attr1 = attr1
        #: The second attribute, a string.
        self.attr2 = attr2

    def example_method(self) -> None:
        """
        This is an example method.
        """

        # Method implementation
        pass
```

## Inline comment style

Use inline comments to clarify complex or non-obvious code sections. Comments should be concise and informative.

Use `#%%` to break up logical blocks of code, especially in longer scripts or notebooks. This is particularly helpful for interactive development and debugging.

Example:

```python
# This is an inline comment explaining a specific part of the code.
x = y + z  # Another inline comment

#%% A new logical block

# Code for the new block...
```

## Whitespace conventions

Consistent use of whitespace improves readability.

*   **Indentation:** Use four spaces for indentation. Do not use tabs.
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

    result = param1 + param2  # Spaces around operator
    my_list = [1, 2, 3]     # Space after comma, no space inside brackets
    return result

# Two blank lines before this function definition

def another_function():
    """
    Another example function.
    """

    # One blank line before this block
    if x > 5:
        print("x is greater than 5")
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

*   **Grouping:** Group imports in the following order:
    1.  Future imports (e.g., `from __future__ import annotations`)
    2.  Standard library imports (e.g., `import os`, `import sys`)
    3.  Third-party library imports (e.g., `import numpy as np`, `import tensorflow as tf`)
    4.  Local application/library specific imports (e.g., `from . import utils`, `from megadetector.detection import run_detector`)
    Separate each group with a blank line.
*   **`from __future__ import annotations`:** Include this import at the beginning of all Python files to enable postponed evaluation of type hints (PEP 563). This allows using type hints for classes defined later in the file.

Example:

```python
from __future__ import annotations

import os
import json
from typing import List, Dict, Optional

import numpy as np
from PIL import Image

from megadetector.utils import some_utility_function
```

## Type hinting

Type hinting is strongly encouraged for all new code to improve code clarity and help with static analysis. While not strictly enforced for retroactive changes to older code, it is highly recommended to add type hints when modifying existing code.

Use standard Python type hints (PEP 484).

Example:

```python
def process_data(data: List[Dict[str, any]], threshold: float = 0.5) -> Optional[str]:
    """
    Processes a list of data dictionaries.

    Args:
        data: A list of dictionaries.
        threshold: A float threshold for processing.

    Returns:
        An optional string, or None if processing fails.
    """

    if not data:
        return None
    # ... processing logic ...
    return "Processed"
```

This document serves as a guide to maintain a high quality and consistent codebase. Please refer to it regularly and apply these conventions in your contributions.

## Linting with Ruff

To help enforce these coding conventions, we use [Ruff](https://docs.astral.sh/ruff/), an extremely fast Python linter and code formatter. Ruff can quickly identify and often automatically fix deviations from our defined style.

### Installation

To install Ruff, run the following command in your terminal:

```bash
pip install ruff
```

### Checking for violations

To check the entire codebase for any linting violations, navigate to the root of the repository and run:

```bash
ruff check .
```

If you want to check a specific file or directory, you can provide its path:

```bash
ruff check path/to/your/file_or_directory
```

### Automatically fixing violations

Ruff can attempt to automatically fix many of the violations it finds. To do this for the entire codebase, run:

```bash
ruff check . --fix
```

Or, for a specific file or directory:

```bash
ruff check path/to/your/file_or_directory --fix
```

### Before committing

It is highly recommended to run the linter (preferably with auto-fixing) on your changes before committing your code. This helps ensure that all contributions maintain the same coding standards and reduces the need for style-related corrections during code reviews.
```
