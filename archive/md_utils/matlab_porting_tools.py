########
#
# matlab_porting_tools.py
#
# Module containing a few ported Matlab functions that made it easier
# for me to port other, larger Matlab functions.
#
########

#%% sec2hms()
        
def sec2hms(tSeconds):
    """
    function [str,h,m,s] = sec2hms(tSeconds,separator)
    
        Convert a time in seconds to a string of the form:
    
        1 hour, 2 minutes, 31.4 seconds
    
    I prefer using the humanfriendly package for this, but I use this when
    porting from Matlab.    
    """

    # https://stackoverflow.com/questions/775049/python-time-seconds-to-hms    
    m, s = divmod(tSeconds, 60)
    h, m = divmod(m, 60)
    
    # colonString = '%d:%02d:%02d' % (h, m, s)
    # return (colonString,verboseString)
    
    hms = ''
    separator = ', '
    if (h > 0):
        pluralString = ''    
        if (h > 1):
            pluralString = 's'
        hms = hms + '%d hour%s%s' % (h,pluralString,separator)
    
    if (m > 0):
        pluralString = ''
        if (m > 1):
            pluralString = 's'
        hms = hms + '%d min%s%s' % (m,pluralString,separator)
    
    hms = hms + '%3.3fsec' % s
            
    return hms

#%% Test driver for sec2hms()
    
if False:

    pass

    #%%
    
    TEST_VALUES = [60033, 30.4, 245234523454.1]
    
    for n in TEST_VALUES:
        s = sec2hms(n)
        print('{} - {}'.format(n,s))


#%% read_lines_from_file()

def read_lines_from_file(filename):
    
    with open(filename) as f:
        content = f.readlines()
    
    # Remove trailing newlines
    content = [x.rstrip() for x in content] 
    
    return content


#%% write_lines_to_file()
    
def write_lines_to_file(lines, filename):
    
    with open(filename,'w') as f:
        for line in lines:
            f.write(line+ '\n')
            

#%% string_ends_with()
            
def string_ends_with(s,query):    
    return s.endswith(query)

def string_starts_with(s,query):    
    return s.startswith(query)
