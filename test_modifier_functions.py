# Test file for docstring_modifier.py (AST-guided string manipulation version)

# Comment before a function that should be ignored
def _ignored_function(param1: int):
    """
    This function should be ignored.
    Args:
        param1 (int): A parameter.
    """
    # Comment inside ignored function
    pass
# Comment after ignored function

def function_needing_optional_added(param1: str, param2: int = 0, param3: bool = True):
    """
    A function to test adding 'optional'.

    Args:
        param1 (str): The first parameter.
        param2 (int, optional): The second parameter, has default. # Needs optional
        param3 (bool, optional): The third parameter, also has default. # Needs optional
    """
    # Comment inside function_needing_optional_added
    return f"{param1}, {param2}, {param3}"

# Comment block between functions
# Line 2
def function_needing_optional_removed(param1: str, param2: int): # Comment on function signature line
    """A function to test removing 'optional'.

    Args:
        param1 (str): The first parameter, no default. # Has optional, should be removed
        param2 (int): The second parameter, also no default. # Has optional, should be removed
    """
    return f"{param1}, {param2}"

# Corrected order of parameters here
def function_with_mixed_optional_issues(param_no_default_has_optional: list, param_default_needs_optional: float = 0.0, param_correct_optional: str = "hello"):
    """Docstring starts on same line as quotes.
    Mixed optional cases.

    Args:
        param_no_default_has_optional (list): No default, but doc has optional (test removal).
        param_default_needs_optional (float, optional): Has default, but doc is missing optional (test addition).
        param_correct_optional (str, optional): Correctly marked.
    """
    pass # Comment at end of function

def function_with_missing_types(param_ok: str, param_missing_type, param_default_missing_type = 1):
    # Comment before docstring
    """
    Function with some types missing in docstring.

    Args:
        param_ok (str): This one is okay.
        param_missing_type: This one is missing its type.
        param_default_missing_type: Missing type, has default. # Type is empty
    """
    # Comment after docstring
    pass

def function_params_missing_from_docstring(param1: int, param2: str, param3: bool = False):
    """Some params are not in the docstring. Args:
        param1 (int): Documented parameter.
    """
    # param2 and param3 are missing from Args section
    pass

def function_params_extra_in_docstring(param1: int):
    """Docstring has params not in signature.
    Args:
        param1 (int): A real parameter.
        extra_param_in_doc (str): This is not in the signature.
        another_extra (bool, optional): Also not in signature.
    """
    pass

def function_no_docstring(param1: int, param2: str):
    # This function has no docstring at all.
    # So all its params should be reported as missing from docstring.
    pass

def function_already_correct(param1: str, param2: int = 5):
    """
    This function's docstring is already correct.

    Args:
        param1 (str): The first parameter.
        param2 (int, optional): The second parameter.
    """
    # Another comment
    return f"{param1} {param2}"

# New test cases for misplaced "optional" and varied spacing
# Case 1: optional at start, sig has default
def test_opt_first_default(param_a: str = "default"):
    """
    Args:
        param_a (str, optional): Description
    """
    pass

# Case 2: optional at end, sig has default (already covered by function_already_correct essentially)

# Case 3: optional at start, sig no default
def test_opt_first_no_default(param_b: str):
    """
    Args:
        param_b (str): Description
    """
    pass

# Case 4: type only, sig has default (covered by function_needing_optional_added)

# Case 5: type only, sig no default (standard, no change)
def test_type_only_no_default(param_c: str):
    """
    Args:
        param_c (str): Description
    """
    pass

# Case 6: optional with complex type and spacing, sig has default
def test_complex_spacing_default(param_d: list or str = None):
    """
    Args:
        param_d (list or str, optional): Description for d
    """
    pass

# Case 7: optional with complex type and spacing, sig no default
def test_complex_spacing_no_default(param_e: list or str):
    """
    Args:
        param_e (list or str): Description for e
    """
    pass

# Case 8: Type that is just "optional", sig has default
def test_only_optional_default(param_f: str = "val"):
    """
    Args:
        param_f: Description for f
    """
    pass

# Case 9: Type that is just "optional", sig no default
def test_only_optional_no_default(param_g: str):
    """
    Args:
        param_g: Description for g
    """
    pass


# Comment before class
class MyTestClass: # Comment on class definition line
    # Comment inside class, before method
    def _ignored_method(self, data:dict):
        """
        This method should be ignored.
        Args:
            data (dict): some data.
        """
        pass

    # Comment before method
    def method_needs_optional_added(self, name: str, age: int = 30):
        """
        A method to test adding 'optional'.

        Args:
            name (str): Name of person.
            age (int, optional): Age of person. # Needs optional
        """
        pass # Comment inside method_needs_optional_added

    def method_needs_optional_removed(self, item_id: str, quantity: int):
        """A method to test removing 'optional'. Args:
            item_id (str): ID of the item. # Remove optional
            quantity (int): Quantity of the item. # Remove optional
        """
        pass

    @classmethod
    def cm_needing_optional(cls, config_path: str, verbose: bool = False): # Test @classmethod
        """
        A classmethod needing optional.

        Args:
            config_path (str): Path to config.
            verbose (bool, optional): Verbosity flag. # Needs optional
        """
        pass

# Comment after class
def function_with_multiline_description(param1: str, param2: int = 0):
    """
    Test multiline descriptions and careful modification.

    This is a line before Args.
    Args:
        param1 (str): This is the first parameter.
            Its description spans multiple lines.
            And should be preserved.
        param2 (int, optional): This is the second. # Needs optional
            It also has a multi-line description.
            And needs optional. This part of desc.
    This is a line after Args.
    """
    pass

def function_with_weird_spacing_in_type(param1: str, param2: int = 0): # param2 sig has default
    """
    Test weird spacing for type and optional marker.

    Args:
        param1 ( str ): Type with spaces. (Should not be changed)
        param2 (int, optional): Optional with no space before comma. (Sig has default, so should be 'int, optional')
    """
    pass

def no_args_section(param1: str):
    """
    This docstring exists but has no Args section.
    So param1 should be reported as missing.
    """
    pass

def empty_args_section(param1: str):
    """
    This docstring has an empty Args section.
    Args:
    """
    # param1 should be reported as missing
    pass

# Fin.
