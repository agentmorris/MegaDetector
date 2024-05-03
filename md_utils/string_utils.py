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
    
    if (size[-1] == 'B'):
        size = size[:-1]
        
    if (size.isdigit()):
        bytes = int(size)
    elif (is_float(size)):
        bytes = float(size)
    else:
        bytes = size[:-1]
        unit = size[-1]
        try:        
            bytes = float(bytes)
            if (unit == 'T'):
                bytes *= 1024*1024*1024*1024
            elif (unit == 'G'):
                bytes *= 1024*1024*1024
            elif (unit == 'M'):
                bytes *= 1024*1024
            elif (unit == 'K'):
                bytes *= 1024
            else:
                bytes = 0
        except ValueError:
            bytes = 0
            
    return bytes


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
