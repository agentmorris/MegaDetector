"""

path_utils.py

Miscellaneous useful utils for path manipulation, i.e. things that could *almost*
be in os.path, but aren't.

"""

#%% Imports and constants

import glob
import ntpath
import os
import sys
import platform
import string
import json
import shutil
import unicodedata
import zipfile
import tarfile
import webbrowser
import subprocess
import re

from zipfile import ZipFile
from datetime import datetime
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from shutil import which
from tqdm import tqdm

# Should all be lower-case
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')

VALID_FILENAME_CHARS = f"~-_.() {string.ascii_letters}{string.digits}"
SEPARATOR_CHARS = r":\/"
VALID_PATH_CHARS = VALID_FILENAME_CHARS + SEPARATOR_CHARS
CHAR_LIMIT = 255


#%% General path functions

def recursive_file_list(base_dir, 
                        convert_slashes=True, 
                        return_relative_paths=False, 
                        sort_files=True,
                        recursive=True):
    r"""
    Enumerates files (not directories) in [base_dir], optionally converting
    backslahes to slashes
    
    Args:
        base_dir (str): folder to enumerate
        convert_slashes (bool, optional): force forward slashes; if this is False, will use
            the native path separator
        return_relative_paths (bool, optional): return paths that are relative to [base_dir],
            rather than absolute paths
        sort_files (bool, optional): force files to be sorted, otherwise uses the sorting
            provided by os.walk()
        recursive (bool, optional): enumerate recursively
        
    Returns:
        list: list of filenames
    """
    
    assert os.path.isdir(base_dir), '{} is not a folder'.format(base_dir)
    
    all_files = []

    if recursive:
        for root, _, filenames in os.walk(base_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                all_files.append(full_path)
    else:
        all_files_relative = os.listdir(base_dir)
        all_files = [os.path.join(base_dir,fn) for fn in all_files_relative]
        all_files = [fn for fn in all_files if os.path.isfile(fn)]
        
    if return_relative_paths:
        all_files = [os.path.relpath(fn,base_dir) for fn in all_files]

    if convert_slashes:
        all_files = [fn.replace('\\', '/') for fn in all_files]
    
    if sort_files:
        all_files = sorted(all_files)
        
    return all_files


def file_list(base_dir, convert_slashes=True, return_relative_paths=False, sort_files=True, 
              recursive=False):
    """
    Trivial wrapper for recursive_file_list, which was a poor function name choice at the time, 
    since it doesn't really make sense to have a "recursive" option in a function called 
    "recursive_file_list".
    
    Args:
        base_dir (str): folder to enumerate
        convert_slashes (bool, optional): force forward slashes; if this is False, will use
            the native path separator
        return_relative_paths (bool, optional): return paths that are relative to [base_dir],
            rather than absolute paths
        sort_files (bool, optional): force files to be sorted, otherwise uses the sorting
            provided by os.walk()
        recursive (bool, optional): enumerate recursively
        
    Returns:
        list: list of filenames    
    """
    
    return recursive_file_list(base_dir,convert_slashes,return_relative_paths,sort_files,
                               recursive=recursive)


def fileparts(path):
    r"""
    Breaks down a path into the directory path, filename, and extension.

    Note that the '.' lives with the extension, and separators are removed.

    Examples:
        
    .. code-block:: none

        >>> fileparts('file')    
        ('', 'file', '')
        >>> fileparts(r'c:/dir/file.jpg')
        ('c:/dir', 'file', '.jpg')
        >>> fileparts('/dir/subdir/file.jpg')
        ('/dir/subdir', 'file', '.jpg')        

    Args:
        path (str): path name to separate into parts
    Returns:
        tuple: tuple containing (p,n,e):        
            - p: str, directory path
            - n: str, filename without extension
            - e: str, extension including the '.'
    """
    
    # ntpath seems to do the right thing for both Windows and Unix paths
    p = ntpath.dirname(path)
    basename = ntpath.basename(path)
    n, e = ntpath.splitext(basename)
    return p, n, e


def insert_before_extension(filename, s=None, separator='.'):
    """
    Insert string [s] before the extension in [filename], separated with [separator].

    If [s] is empty, generates a date/timestamp. If [filename] has no extension,
    appends [s].

    Examples:
        
    .. code-block:: none
    
        >>> insert_before_extension('/dir/subdir/file.ext', 'insert')
        '/dir/subdir/file.insert.ext'
        >>> insert_before_extension('/dir/subdir/file', 'insert')
        '/dir/subdir/file.insert'
        >>> insert_before_extension('/dir/subdir/file')
        '/dir/subdir/file.2020.07.20.10.54.38'
        
    Args:
        filename (str): filename to manipulate
        s (str, optional): string to insert before the extension in [filename], or
            None to insert a datestamp
        separator (str, optional): separator to place between the filename base
            and the inserted string
            
    Returns:
        str: modified string
    """
    
    assert len(filename) > 0
    if s is None or len(s) == 0:
        s = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    name, ext = os.path.splitext(filename)
    return f'{name}{separator}{s}{ext}'


def split_path(path):
    r"""
    Splits [path] into all its constituent file/folder tokens.

    Examples:
        
    .. code-block:: none
    
        >>> split_path(r'c:\dir\subdir\file.txt')
        ['c:\\', 'dir', 'subdir', 'file.txt']
        >>> split_path('/dir/subdir/file.jpg')
        ['/', 'dir', 'subdir', 'file.jpg']
        >>> split_path('c:\\')
        ['c:\\']
        >>> split_path('/')
        ['/']
        
    Args:
        path (str): path to split into tokens
    
    Returns:
        list: list of path tokens
    """
    
    parts = []
    while True:
        # ntpath seems to do the right thing for both Windows and Unix paths
        head, tail = ntpath.split(path)
        if head == '' or head == path:
            break
        parts.append(tail)
        path = head
    parts.append(head or tail)
    return parts[::-1] # reverse


def path_is_abs(p):
    """
    Determines whether [p] is an absolute path.  An absolute path is defined as
    one that starts with slash, backslash, or a letter followed by a colon.
    
    Args:
        p (str): path to evaluate
        
    Returns:
        bool: True if [p] is an absolute path, else False
    """
    
    return (len(p) > 1) and (p[0] == '/' or p[1] == ':' or p[0] == '\\')


def top_level_folder(p):
    r"""
    Gets the top-level folder from the path *p*.
    
    On UNIX, this is straightforward:
        
    /blah/foo 
    
    ...returns '/blah'
    
    On Windows, we define this as the top-level folder that isn't the drive, so:
        
    c:\blah\foo
    
    ...returns 'c:\blah'.
    
    Args:
        p (str): filename to evaluate
        
    Returns:
        str: the top-level folder in [p], see above for details on how this is defined
    """
    
    if p == '':
        return ''
    
    # Path('/blah').parts is ('/','blah')
    parts = split_path(p)
    
    if len(parts) == 1:
        return parts[0]

    # Handle paths like:
    #
    # /, \, /stuff, c:, c:\stuff
    drive = os.path.splitdrive(p)[0]
    if parts[0] == drive or parts[0] == drive + '/' or parts[0] == drive + '\\' or parts[0] in ['\\', '/']:
        return os.path.join(parts[0], parts[1])
    else:
        return parts[0]

# ...top_level_folder()


#%% Test driver for top_level_folder

if False:  

    #%%
    
    p = 'blah/foo/bar'; s = top_level_folder(p); print(s); assert s == 'blah'
    p = '/blah/foo/bar'; s = top_level_folder(p); print(s); assert s == '/blah'
    p = 'bar'; s = top_level_folder(p); print(s); assert s == 'bar'
    p = ''; s = top_level_folder(p); print(s); assert s == ''
    p = 'c:\\'; s = top_level_folder(p); print(s); assert s == 'c:\\'
    p = r'c:\blah'; s = top_level_folder(p); print(s); assert s == 'c:\\blah'
    p = r'c:\foo'; s = top_level_folder(p); print(s); assert s == 'c:\\foo'
    p = r'c:/foo'; s = top_level_folder(p); print(s); assert s == 'c:/foo'
    p = r'c:\foo/bar'; s = top_level_folder(p); print(s); assert s == 'c:\\foo'
        
    #%%

def safe_create_link(link_exists,link_new):
    """
    Creates a symlink at [link_new] pointing to [link_exists].
    
    If [link_new] already exists, make sure it's a link (not a file),
    and if it has a different target than [link_exists], removes and re-creates
    it.
    
    Errors if [link_new] already exists but it's not a link.
    
    Args:
        link_exists (str): the source of the (possibly-new) symlink
        link_new (str): the target of the (possibly-new) symlink
    """
    
    if os.path.exists(link_new) or os.path.islink(link_new):
        assert os.path.islink(link_new)
        if not os.readlink(link_new) == link_exists:
            os.remove(link_new)
            os.symlink(link_exists,link_new)
    else:
        os.symlink(link_exists,link_new)
        

#%% Image-related path functions

def is_image_file(s, img_extensions=IMG_EXTENSIONS):
    """
    Checks a file's extension against a hard-coded set of image file
    extensions.  Uses case-insensitive comparison.
    
    Does not check whether the file exists, only determines whether the filename
    implies it's an image file.
    
    Args:
        s (str): filename to evaluate for image-ness
        img_extensions (list, optional): list of known image file extensions
        
    Returns:
        bool: True if [s] appears to be an image file, else False
    """
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in img_extensions


def find_image_strings(strings):
    """
    Given a list of strings that are potentially image file names, looks for
    strings that actually look like image file names (based on extension).
    
    Args:
        strings (list): list of filenames to check for image-ness
        
    Returns:
        list: the subset of [strings] that appear to be image filenames
    """
    
    return [s for s in strings if is_image_file(s)]


def find_images(dirname, 
                recursive=False, 
                return_relative_paths=False, 
                convert_slashes=True):
    """
    Finds all files in a directory that look like image file names. Returns
    absolute paths unless return_relative_paths is set.  Uses the OS-native
    path separator unless convert_slashes is set, in which case will always
    use '/'.
    
    Args:
        dirname (str): the folder to search for images
        recursive (bool, optional): whether to search recursively
        return_relative_paths (str, optional): return paths that are relative
            to [dirname], rather than absolute paths
        convert_slashes (bool, optional): force forward slashes in return values

    Returns:
        list: list of image filenames found in [dirname]
    """
    
    assert os.path.isdir(dirname), '{} is not a folder'.format(dirname)
    
    if recursive:
        strings = glob.glob(os.path.join(dirname, '**', '*.*'), recursive=True)
    else:
        strings = glob.glob(os.path.join(dirname, '*.*'))
    
    image_files = find_image_strings(strings)
    
    if return_relative_paths:
        image_files = [os.path.relpath(fn,dirname) for fn in image_files]
    
    image_files = sorted(image_files)
    
    if convert_slashes:
        image_files = [fn.replace('\\', '/') for fn in image_files]
        
    return image_files


#%% Filename cleaning functions

def clean_filename(filename, 
                   allow_list=VALID_FILENAME_CHARS,
                   char_limit=CHAR_LIMIT,
                   force_lower= False):
    r"""
    Removes non-ASCII and other invalid filename characters (on any
    reasonable OS) from a filename, then optionally trims to a maximum length.

    Does not allow :\/ by default, use clean_path if you want to preserve those.

    Adapted from
    https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
    
    Args:
        filename (str): filename to clean
        allow_list (str, optional): string containing all allowable filename characters
        char_limit (int, optional): maximum allowable filename length, if None will skip this
            step
        force_lower (bool, optional): convert the resulting filename to lowercase
    
    returns:
        str: cleaned version of [filename]            
    """
    
    # keep only valid ascii chars
    cleaned_filename = (unicodedata.normalize('NFKD', filename)
                        .encode('ASCII', 'ignore').decode())

    # keep only allow-listed chars
    cleaned_filename = ''.join([c for c in cleaned_filename if c in allow_list])
    if char_limit is not None:
        cleaned_filename = cleaned_filename[:char_limit]
    if force_lower:
        cleaned_filename = cleaned_filename.lower()
    return cleaned_filename


def clean_path(pathname, 
               allow_list=VALID_PATH_CHARS,
               char_limit=CHAR_LIMIT,
               force_lower=False):
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then optionally trims to a maximum length.
    
    Args:
        pathname (str): path name to clean
        allow_list (str, optional): string containing all allowable filename characters
        char_limit (int, optional): maximum allowable filename length, if None will skip this
            step
        force_lower (bool, optional): convert the resulting filename to lowercase
    
    returns:
        str: cleaned version of [filename]            
    """
    
    return clean_filename(pathname, allow_list=allow_list, 
                          char_limit=char_limit, force_lower=force_lower)


def flatten_path(pathname,separator_chars=SEPARATOR_CHARS,separator_char_replacement='~'):
    r"""
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then trims to a maximum length. Replaces all valid
    separators with [separator_char_replacement.]
    
    Args:
        pathname (str): path name to flatten
        separator_chars (str, optional): string containing all known path separators
        separator_char_replacement (str, optional): string to insert in place of 
            path separators.
            
    Returns:
        str: flattened version of [pathname]
    """
    
    s = clean_path(pathname)
    for c in separator_chars:
        s = s.replace(c, separator_char_replacement)
    return s


def is_executable(filename):    
    """
    Checks whether [filename] is on the system path and marked as executable.
    
    Args:
        filename (str): filename to check for executable status
    
    Returns:
        bool: True if [filename] is on the system path and marked as executable, otherwise False
    """
    
    # https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    return which(filename) is not None


#%% Platform-independent way to open files in their associated application

def environment_is_wsl():
    """
    Determines whether we're running in WSL.
    
    Returns:
        True if we're running in WSL.        
    """
    
    if sys.platform not in ('linux','posix'):
        return False
    platform_string = ' '.join(platform.uname()).lower()
    return 'microsoft' in platform_string and 'wsl' in platform_string
    

def wsl_path_to_windows_path(filename):
    r"""
    Converts a WSL path to a Windows path, or returns None if that's not possible.  E.g.
    converts:
        
    /mnt/e/a/b/c
    
    ...to:
        
    e:\a\b\c
    
    Args:
        filename (str): filename to convert
    
    Returns:
        str: Windows equivalent to the WSL path [filename]
    """
    
    result = subprocess.run(['wslpath', '-w', filename], text=True, capture_output=True)
    if result.returncode != 0:
        print('Could not convert path {} from WSL to Windows'.format(filename))
        return None
    return result.stdout.strip()
    
    
def open_file(filename, attempt_to_open_in_wsl_host=False, browser_name=None):
    """
    Opens [filename] in the default OS file handler for this file type.
    
    If browser_name is not None, uses the webbrowser module to open the filename
    in the specified browser; see https://docs.python.org/3/library/webbrowser.html
    for supported browsers.  Falls back to the default file handler if webbrowser.open()
    fails.  In this case, attempt_to_open_in_wsl_host is ignored unless webbrowser.open() fails.
    
    If browser_name is 'default', uses the system default.  This is different from the 
    parameter to webbrowser.get(), where None implies the system default.
    
    Args:
        filename (str): file to open
        attempt_to_open_in_wsl_host: if this is True, and we're in WSL, attempts to open
            [filename] in the Windows host environment
        browser_name: see above
    """
    
    if browser_name is not None:
        if browser_name == 'chrome':
            browser_name = 'google-chrome'
        elif browser_name == 'default':
            browser_name = None
        try:
            result = webbrowser.get(using=browser_name).open(filename)
        except Exception:
            result = False
        if result:
            return
        
    if sys.platform == 'win32':
        
        os.startfile(filename)

    elif sys.platform == 'darwin':
      
        opener = 'open'
        subprocess.call([opener, filename])
            
    elif attempt_to_open_in_wsl_host and environment_is_wsl():
        
        windows_path = wsl_path_to_windows_path(filename)
        
        # Fall back to xdg-open
        if windows_path is None:
            subprocess.call(['xdg-open', filename])
            
        if os.path.isdir(filename):            
            subprocess.run(["explorer.exe", windows_path])
        else:
            os.system("cmd.exe /C start %s" % (re.escape(windows_path)))    
        
    else:
        
        opener = 'xdg-open'        
        subprocess.call([opener, filename])
        

#%% File list functions

def write_list_to_file(output_file,strings):
    """
    Writes a list of strings to either a JSON file or text file,
    depending on extension of the given file name.
    
    Args:
        output_file (str): file to write
        strings (list): list of strings to write to [output_file]
    """
    
    with open(output_file, 'w') as f:
        if output_file.endswith('.json'):
            json.dump(strings, f, indent=1)
        else:
            f.write('\n'.join(strings))


def read_list_from_file(filename):
    """
    Reads a json-formatted list of strings from a file.
    
    Args:
        filename (str): .json filename to read
    
    Returns:
        list: list of strings read from [filename]
    """
    
    assert filename.endswith('.json')
    with open(filename, 'r') as f:
        file_list = json.load(f)
    assert isinstance(file_list, list)
    for s in file_list:
        assert isinstance(s, str)
    return file_list


def _copy_file(input_output_tuple,overwrite=True,verbose=False):
    """
    Internal function for copying files from within parallel_copy_files.
    """
    
    assert len(input_output_tuple) == 2
    source_fn = input_output_tuple[0]
    target_fn = input_output_tuple[1]
    if (not overwrite) and (os.path.isfile(target_fn)):
        if verbose:
            print('Skipping existing file {}'.format(target_fn))
        return
    os.makedirs(os.path.dirname(target_fn),exist_ok=True)
    shutil.copyfile(source_fn,target_fn)
    

def parallel_copy_files(input_file_to_output_file, max_workers=16, 
                        use_threads=True, overwrite=False, verbose=False):
    """
    Copies files from source to target according to the dict input_file_to_output_file.
    
    Args:
        input_file_to_output_file (dict): dictionary mapping source files to the target files
            to which they should be copied
        max_workers (int, optional): number of concurrent workers; set to <=1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallel copying; ignored if max_workers <= 1
        overwrite (bool, optional): whether to overwrite existing destination files
        verbose (bool, optional): enable additionald debug output
    """

    n_workers = min(max_workers,len(input_file_to_output_file))
    
    # Package the dictionary as a set of 2-tuples
    input_output_tuples = []
    for input_fn in input_file_to_output_file:
        input_output_tuples.append((input_fn,input_file_to_output_file[input_fn]))

    if use_threads:
        pool = ThreadPool(n_workers)
    else:
        pool = Pool(n_workers)

    with tqdm(total=len(input_output_tuples)) as pbar:
        for i,_ in enumerate(pool.imap_unordered(partial(_copy_file,overwrite=overwrite,verbose=verbose),
                                                 input_output_tuples)):
            pbar.update()

# ...def parallel_copy_files(...)


def get_file_sizes(base_dir, convert_slashes=True):
    """
    Gets sizes recursively for all files in base_dir, returning a dict mapping
    relative filenames to size.
    
    TODO: merge the functionality here with parallel_get_file_sizes, which uses slightly
    different semantics.
    
    Args:
        base_dir (str): folder within which we want all file sizes
        convert_slashes (bool, optional): force forward slashes in return strings,
            otherwise uses the native path separator
            
    Returns:
        dict: dictionary mapping filenames to file sizes in bytes
    """
    
    relative_filenames = recursive_file_list(base_dir, convert_slashes=convert_slashes, 
                                             return_relative_paths=True)
    
    fn_to_size = {}
    for fn_relative in tqdm(relative_filenames):
        fn_abs = os.path.join(base_dir,fn_relative)
        fn_to_size[fn_relative] = os.path.getsize(fn_abs)
                   
    return fn_to_size
        

def _get_file_size(filename,verbose=False):
    """
    Internal function for safely getting the size of a file.  Returns a (filename,size)
    tuple, where size is None if there is an error.
    """
    
    try:
        size = os.path.getsize(filename)
    except Exception as e:
        if verbose:
            print('Error reading file size for {}: {}'.format(filename,str(e)))
        size = None
    return (filename,size)

    
def parallel_get_file_sizes(filenames, 
                            max_workers=16, 
                            use_threads=True, 
                            verbose=False,
                            recursive=True, 
                            convert_slashes=True,
                            return_relative_paths=False):
    """
    Returns a dictionary mapping every file in [filenames] to the corresponding file size,
    or None for errors.  If [filenames] is a folder, will enumerate the folder (optionally recursively).
        
    Args:
        filenames (list or str): list of filenames for which we should read sizes, or a folder
            within which we should read all file sizes recursively
        max_workers (int, optional): number of concurrent workers; set to <=1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallel copying; ignored if max_workers <= 1
        verbose (bool, optional): enable additionald debug output
        recursive (bool, optional): enumerate recursively, only relevant if [filenames] is a folder.
        convert_slashes (bool, optional): convert backslashes to forward slashes
        return_relative_paths (bool, optional): return relative paths; only relevant if [filenames]
            is a folder.
        
    Returns:
        dict: dictionary mapping filenames to file sizes in bytes
    """

    n_workers = min(max_workers,len(filenames))
    
    folder_name = None
    
    if verbose:
        print('Enumerating files')
        
    if isinstance(filenames,str) and os.path.isdir(filenames):
        
        folder_name = filenames
        
        # Enumerate absolute paths here, we'll convert to relative later if requested
        filenames = recursive_file_list(filenames,recursive=recursive,return_relative_paths=False)

    if verbose:
        print('Creating worker pool')
    
    if use_threads:
        pool_string = 'thread'
        pool = ThreadPool(n_workers)
    else:
        pool_string = 'process'
        pool = Pool(n_workers)

    if verbose:
        print('Created a {} pool of {} workers'.format(
            pool_string,n_workers))
        
    # This returns (filename,size) tuples
    get_size_results = list(tqdm(pool.imap(
        partial(_get_file_size,verbose=verbose),filenames), total=len(filenames)))
    
    to_return = {}
    for r in get_size_results:
        fn = r[0]
        if return_relative_paths and (folder_name is not None):
            fn = os.path.relpath(fn,folder_name)
        if convert_slashes:
            fn = fn.replace('\\','/')
        size = r[1]
        to_return[fn] = size

    return to_return


#%% Zip functions

def zip_file(input_fn, output_fn=None, overwrite=False, verbose=False, compresslevel=9):
    """
    Zips a single file.
    
    Args:
        input_fn (str): file to zip
        output_fn (str, optional): target zipfile; if this is None, we'll use
            [input_fn].zip
        overwrite (bool, optional): whether to overwrite an existing target file
        verbose (bool, optional): enable existing debug console output
        compresslevel (int, optional): compression level to use, between 0 and 9
        
    Returns:
        str: the output zipfile, whether we created it or determined that it already exists
    """
    
    basename = os.path.basename(input_fn)
    
    if output_fn is None:
        output_fn = input_fn + '.zip'
        
    if (not overwrite) and (os.path.isfile(output_fn)):
        print('Skipping existing file {}'.format(output_fn))
        return output_fn
    
    if verbose:
        print('Zipping {} to {} with level {}'.format(input_fn,output_fn,compresslevel))
    
    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_fn,arcname=basename,compresslevel=compresslevel,
                   compress_type=zipfile.ZIP_DEFLATED)

    return output_fn


def add_files_to_single_tar_file(input_files, output_fn, arc_name_base,
                                 overwrite=False, verbose=False, mode='x'):
    """
    Adds all the files in [input_files] to the tar file [output_fn].  
    Archive names are relative to arc_name_base.
    
    Args:
        input_files (list): list of absolute filenames to include in the .tar file
        output_fn (str): .tar file to create
        arc_name_base (str): absolute folder from which relative paths should be determined;
            behavior is undefined if there are files in [input_files] that don't live within
            [arc_name_base]
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
        mode (str, optional): compression type, can be 'x' (no compression), 'x:gz', or 'x:bz2'.
        
    Returns:
        str: the output tar file, whether we created it or determined that it already exists
    """
    
    if os.path.isfile(output_fn):
        if not overwrite:
            print('Tar file {} exists, skipping'.format(output_fn))
            return output_fn
        else:
            print('Tar file {} exists, deleting and re-creating'.format(output_fn))
            os.remove(output_fn)
                
    if verbose:
        print('Adding {} files to {} (mode {})'.format(
            len(input_files),output_fn,mode))
        
    with tarfile.open(output_fn,mode) as tarf:
        for input_fn_abs in tqdm(input_files,disable=(not verbose)):
            input_fn_relative = os.path.relpath(input_fn_abs,arc_name_base)
            tarf.add(input_fn_abs,arcname=input_fn_relative)

    return output_fn


def zip_files_into_single_zipfile(input_files, output_fn, arc_name_base,
                                  overwrite=False, verbose=False, compresslevel=9):
    """
    Zip all the files in [input_files] into [output_fn].  Archive names are relative to 
    arc_name_base.
    
    Args:
        input_files (list): list of absolute filenames to include in the .tar file
        output_fn (str): .tar file to create
        arc_name_base (str): absolute folder from which relative paths should be determined;
            behavior is undefined if there are files in [input_files] that don't live within
            [arc_name_base]
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
        compresslevel (int, optional): compression level to use, between 0 and 9
        
    Returns:
        str: the output zipfile, whether we created it or determined that it already exists
    """
    
    if not overwrite:
        if os.path.isfile(output_fn):
            print('Zip file {} exists, skipping'.format(output_fn))
            return output_fn
        
    if verbose:
        print('Zipping {} files to {} (compression level {})'.format(
            len(input_files),output_fn,compresslevel))
        
    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        for input_fn_abs in tqdm(input_files,disable=(not verbose)):
            input_fn_relative = os.path.relpath(input_fn_abs,arc_name_base)
            zipf.write(input_fn_abs,
                       arcname=input_fn_relative,
                       compresslevel=compresslevel,
                       compress_type=zipfile.ZIP_DEFLATED)

    return output_fn
    
    
def zip_folder(input_folder, output_fn=None, overwrite=False, verbose=False, compresslevel=9):
    """
    Recursively zip everything in [input_folder] into a single zipfile, storing files as paths 
    relative to [input_folder].
    
    Args: 
        input_folder (str): folder to zip
        output_fn (str, optional): output filename; if this is None, we'll write to [input_folder].zip
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
        compresslevel (int, optional): compression level to use, between 0 and 9
        
    Returns:
        str: the output zipfile, whether we created it or determined that it already exists    
    """
    
    if output_fn is None:
        output_fn = input_folder + '.zip'
        
    if not overwrite:
        if os.path.isfile(output_fn):
            print('Zip file {} exists, skipping'.format(output_fn))
            return            
        
    if verbose:
        print('Zipping {} to {} (compression level {})'.format(
            input_folder,output_fn,compresslevel))
    
    relative_filenames = recursive_file_list(input_folder,return_relative_paths=True)
    
    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        for input_fn_relative in tqdm(relative_filenames,disable=(not verbose)):
            input_fn_abs = os.path.join(input_folder,input_fn_relative)            
            zipf.write(input_fn_abs,
                       arcname=input_fn_relative,
                       compresslevel=compresslevel,
                       compress_type=zipfile.ZIP_DEFLATED)

    return output_fn

        
def parallel_zip_files(input_files, max_workers=16, use_threads=True, compresslevel=9, 
                       overwrite=False, verbose=False):
    """
    Zips one or more files to separate output files in parallel, leaving the 
    original files in place.  Each file is zipped to [filename].zip.
    
    Args:
        input_file (str): list of files to zip
        max_workers (int, optional): number of concurrent workers, set to <= 1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False); ignored if
            max_workers <= 1
        compresslevel (int, optional): zip compression level between 0 and 9
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
    """

    n_workers = min(max_workers,len(input_files))

    if use_threads:
        pool = ThreadPool(n_workers)
    else:
        pool = Pool(n_workers)

    with tqdm(total=len(input_files)) as pbar:
        for i,_ in enumerate(pool.imap_unordered(partial(zip_file,
          output_fn=None,overwrite=overwrite,verbose=verbose,compresslevel=compresslevel),
          input_files)):
            pbar.update()


def parallel_zip_folders(input_folders, max_workers=16, use_threads=True,
                         compresslevel=9, overwrite=False, verbose=False):
    """
    Zips one or more folders to separate output files in parallel, leaving the 
    original folders in place.  Each folder is zipped to [folder_name].zip.
    
    Args:
        input_folder (list): list of folders to zip
        max_workers (int, optional): number of concurrent workers, set to <= 1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False); ignored if
            max_workers <= 1
        compresslevel (int, optional): zip compression level between 0 and 9
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
    """

    n_workers = min(max_workers,len(input_folders))

    if use_threads:
        pool = ThreadPool(n_workers)
    else:
        pool = Pool(n_workers)
    
    with tqdm(total=len(input_folders)) as pbar:
        for i,_ in enumerate(pool.imap_unordered(
                partial(zip_folder,overwrite=overwrite,
                        compresslevel=compresslevel,verbose=verbose),
                input_folders)):
            pbar.update()


def zip_each_file_in_folder(folder_name,recursive=False,max_workers=16,use_threads=True,
                            compresslevel=9,overwrite=False,required_token=None,verbose=False,
                            exclude_zip=True):
    """
    Zips each file in [folder_name] to its own zipfile (filename.zip), optionally recursing.  To 
    zip a whole folder into a single zipfile, use zip_folder().
    
    Args:
        folder_name (str): the folder within which we should zip files
        recursive (bool, optional): whether to recurse within [folder_name]
        max_workers (int, optional): number of concurrent workers, set to <= 1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False); ignored if
            max_workers <= 1
        compresslevel (int, optional): zip compression level between 0 and 9
        overwrite (bool, optional): whether to overwrite an existing .tar file
        required_token (str, optional): only zip files whose names contain this string
        verbose (bool, optional): enable additional debug console output
        exclude_zip (bool, optional): skip files ending in .zip        
    """
    
    assert os.path.isdir(folder_name), '{} is not a folder'.format(folder_name)
    
    input_files = recursive_file_list(folder_name,recursive=recursive,return_relative_paths=False)
    
    if required_token is not None:
        input_files = [fn for fn in input_files if required_token in fn]
    
    if exclude_zip:
        input_files = [fn for fn in input_files if (not fn.endswith('.zip'))]
                                                    
    parallel_zip_files(input_files=input_files,max_workers=max_workers,
                       use_threads=use_threads,compresslevel=compresslevel,
                       overwrite=overwrite,verbose=verbose)


def unzip_file(input_file, output_folder=None):
    """
    Unzips a zipfile to the specified output folder, defaulting to the same location as
    the input file.
    
    Args:
        input_file (str): zipfile to unzip
        output_folder (str, optional): folder to which we should unzip [input_file], defaults
            to unzipping to the folder where [input_file] lives
    """
    
    if output_folder is None:
        output_folder = os.path.dirname(input_file)
        
    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)
