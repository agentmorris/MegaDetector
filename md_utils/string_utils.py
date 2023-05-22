########
#
# string_utils.py
#
# Miscellaneous string utilities
#
########

import re

def is_float(s):
    """ 
    Checks whether a string represents a valid float
    """
    
    try:
        _ = float(s)
    except ValueError:
        return False
    return True


def human_readable_to_bytes(size):
    """
    Given a human-readable byte string (e.g. 2G, 10GB, 30MB, 20KB),
    return the number of bytes.  Will return 0 if the argument has
    unexpected form.
    
    https://gist.github.com/beugley/ccd69945346759eb6142272a6d69b4e0
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
