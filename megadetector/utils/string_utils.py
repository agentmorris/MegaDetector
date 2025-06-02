"""

string_utils.py

Miscellaneous string utilities.

"""

#%% Imports

import re


#%% Functions

def is_float(s):
    """
    Checks whether [s] is an object (typically a string) that can be cast to a float

    Args:
        s (object): object to evaluate

    Returns:
        bool: True if s successfully casts to a float, otherwise False
    """

    if s is None:
        return False

    try:
        _ = float(s)
    except ValueError:
        return False
    return True


def human_readable_to_bytes(size):
    """
    Given a human-readable byte string (e.g. 2G, 10GB, 30MB, 20KB),
    returns the number of bytes.  Will return 0 if the argument has
    unexpected form.

    https://gist.github.com/beugley/ccd69945346759eb6142272a6d69b4e0

    Args:
        size (str): string representing a size

    Returns:
        int: the corresponding size in bytes
    """

    size = re.sub(r'\s+', '', size)

    if not size: # Handle empty string case after stripping spaces
        return 0

    if (size[-1] == 'B'):
        size = size[:-1]

    if not size: # Handle case where size was just "B"
        return 0

    if (size.isdigit()):
        bytes_val = int(size) # Renamed to avoid conflict with built-in 'bytes'
    elif (is_float(size)):
        bytes_val = float(size) # Renamed
    else:
        # Handle cases like "1KB" where size[:-1] might be "1K" before this block
        # The original code would try to float("1K") which fails.
        # Need to separate numeric part from unit more carefully.
        numeric_part = ''
        unit_part = ''

        # Iterate from the end to find the unit (K, M, G, T)
        # This handles cases like "10KB" or "2.5GB"
        for i in range(len(size) -1, -1, -1):
            if size[i].isalpha():
                unit_part = size[i] + unit_part
            else:
                numeric_part = size[:i+1]
                break

        # If no unit found, or numeric part is empty after stripping unit
        if not unit_part or not numeric_part:
            return 0

        try:
            bytes_val = float(numeric_part)
            unit = unit_part
            if (unit == 'T'):
                bytes_val *= 1024*1024*1024*1024
            elif (unit == 'G'):
                bytes_val *= 1024*1024*1024
            elif (unit == 'M'):
                bytes_val *= 1024*1024
            elif (unit == 'K'):
                bytes_val *= 1024
            else:
                # If it's a known unit (like 'B' already stripped) but not T/G/M/K,
                # and it was floatable, it's just bytes.  If it's an unknown unit, it's
                # an error.
                if unit not in ['B', '']: # 'B' was stripped, '' means just a number
                     bytes_val = 0
        except ValueError:
            bytes_val = 0

    return bytes_val


def remove_ansi_codes(s):
    """
    Removes ANSI escape codes from a string.

    https://stackoverflow.com/questions/14693701/how-can-i-remove-the-ansi-escape-sequences-from-a-string-in-python#14693789

    Args:
        s (str): the string to de-ANSI-i-fy

    Returns:
        str: A copy of [s] without ANSI codes
    """

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', s)


#%% Tests


class TestStringUtils:
    """
    Tests for string_utils.py
    """


    def test_is_float(self):
        """
        Test the is_float function.
        """

        assert is_float("1.23")
        assert is_float("-0.5")
        assert is_float("0")
        assert is_float(1.23)
        assert is_float(0)
        assert not is_float("abc")
        assert not is_float("1.2.3")
        assert not is_float("")
        assert not is_float(None)
        assert not is_float("1,23")


    def test_human_readable_to_bytes(self):
        """
        Test the human_readable_to_bytes function.
        """

        assert human_readable_to_bytes("10B") == 10
        assert human_readable_to_bytes("10") == 10
        assert human_readable_to_bytes("1K") == 1024
        assert human_readable_to_bytes("1KB") == 1024
        assert human_readable_to_bytes("1M") == 1024*1024
        assert human_readable_to_bytes("1MB") == 1024*1024
        assert human_readable_to_bytes("1G") == 1024*1024*1024
        assert human_readable_to_bytes("1GB") == 1024*1024*1024
        assert human_readable_to_bytes("1T") == 1024*1024*1024*1024
        assert human_readable_to_bytes("1TB") == 1024*1024*1024*1024

        assert human_readable_to_bytes("2.5K") == 2.5 * 1024
        assert human_readable_to_bytes("0.5MB") == 0.5 * 1024 * 1024

        # Test with spaces
        assert human_readable_to_bytes(" 2 G ") == 2 * 1024*1024*1024
        assert human_readable_to_bytes("500 KB") == 500 * 1024

        # Invalid inputs
        assert human_readable_to_bytes("abc") == 0
        assert human_readable_to_bytes("1X") == 0
        assert human_readable_to_bytes("1KBB") == 0
        assert human_readable_to_bytes("K1") == 0
        assert human_readable_to_bytes("") == 0
        assert human_readable_to_bytes("1.2.3K") == 0
        assert human_readable_to_bytes("B") == 0


    def test_remove_ansi_codes(self):
        """
        Test the remove_ansi_codes function.
        """

        assert remove_ansi_codes("text without codes") == "text without codes"
        assert remove_ansi_codes("\x1b[31mRed text\x1b[0m") == "Red text"
        assert remove_ansi_codes("\x1b[1m\x1b[4mBold and Underline\x1b[0m") == "Bold and Underline"
        assert remove_ansi_codes("Mixed \x1b[32mgreen\x1b[0m and normal") == "Mixed green and normal"
        assert remove_ansi_codes("") == ""

        # More complex/varied ANSI codes
        assert remove_ansi_codes("text\x1b[1Aup") == "textup"
        assert remove_ansi_codes("\x1b[2Jclearscreen") == "clearscreen"


def test_string_utils():
    """
    Runs all tests in the TestStringUtils class.
    """

    test_instance = TestStringUtils()
    test_instance.test_is_float()
    test_instance.test_human_readable_to_bytes()
    test_instance.test_remove_ansi_codes()

# from IPython import embed; embed()
# test_string_utils()
