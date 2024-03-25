########
# 
# path_utils.py
#
# Miscellaneous useful utils for path manipulation, things that could *almost*
# be in os.path, but aren't.
#
########

#%% Imports and constants

import glob
import ntpath
import os
import sys
import platform
import posixpath
import string
import json
import shutil
import unicodedata
import zipfile
import webbrowser
import subprocess
import re

from zipfile import ZipFile
from datetime import datetime
from typing import Container, Iterable, List, Optional, Tuple, Sequence
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from tqdm import tqdm

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.gif', '.png', '.tif', '.tiff', '.bmp')

VALID_FILENAME_CHARS = f"~-_.() {string.ascii_letters}{string.digits}"
SEPARATOR_CHARS = r":\/"
VALID_PATH_CHARS = VALID_FILENAME_CHARS + SEPARATOR_CHARS
CHAR_LIMIT = 255


#%% General path functions

def recursive_file_list(base_dir, convert_slashes=True, 
                        return_relative_paths=False, sort_files=True,
                        recursive=True):
    r"""
    Enumerate files (not directories) in [base_dir], optionally converting
    \ to /
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
    it doesn't really make sense to have a "recursive" option in a function called "recursive_file_list".
    """
    
    return recursive_file_list(base_dir,convert_slashes,return_relative_paths,sort_files,
                               recursive=recursive)


def split_path(path: str) -> List[str]:
    r"""
    Splits [path] into all its constituent tokens.

    Non-recursive version of:
    http://nicks-liquid-soapbox.blogspot.com/2011/03/splitting-path-to-list-in-python.html

    Examples
    >>> split_path(r'c:\dir\subdir\file.txt')
    ['c:\\', 'dir', 'subdir', 'file.txt']
    >>> split_path('/dir/subdir/file.jpg')
    ['/', 'dir', 'subdir', 'file.jpg']
    >>> split_path('c:\\')
    ['c:\\']
    >>> split_path('/')
    ['/']
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
    return parts[::-1]  # reverse


def fileparts(path: str) -> Tuple[str, str, str]:
    r"""
    Breaks down a path into the directory path, filename, and extension.

    Note that the '.' lives with the extension, and separators are removed.

    Examples
    >>> fileparts('file')
    ('', 'file', '')
    >>> fileparts(r'c:\dir\file.jpg')
    ('c:\\dir', 'file', '.jpg')
    >>> fileparts('/dir/subdir/file.jpg')
    ('/dir/subdir', 'file', '.jpg')

    Returns:
        p: str, directory path
        n: str, filename without extension
        e: str, extension including the '.'
    """
    
    # ntpath seems to do the right thing for both Windows and Unix paths
    p = ntpath.dirname(path)
    basename = ntpath.basename(path)
    n, e = ntpath.splitext(basename)
    return p, n, e


def insert_before_extension(filename: str, s: str = '', separator='.') -> str:
    """
    Insert string [s] before the extension in [filename], separated with [separator].

    If [s] is empty, generates a date/timestamp. If [filename] has no extension,
    appends [s].

    Examples
    >>> insert_before_extension('/dir/subdir/file.ext', 'insert')
    '/dir/subdir/file.insert.ext'
    >>> insert_before_extension('/dir/subdir/file', 'insert')
    '/dir/subdir/file.insert'
    >>> insert_before_extension('/dir/subdir/file')
    '/dir/subdir/file.2020.07.20.10.54.38'
    """
    
    assert len(filename) > 0
    if len(s) == 0:
        s = datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
    name, ext = os.path.splitext(filename)
    return f'{name}{separator}{s}{ext}'


def top_level_folder(p: str, windows: Optional[bool] = None) -> str:
    """
    Gets the top-level folder from path [p].

    This function behaves differently for Windows vs. Unix paths. Set
    windows=True if [p] is a Windows path. Set windows=None (default) to treat
    [p] as a native system path.

    On Windows, will use the top-level folder that isn't the drive.
    >>> top_level_folder(r'c:\blah\foo')
    'c:\blah'

    On Unix, does not include the leaf node.
    >>> top_level_folder('/blah/foo')
    '/blah'
    """
    
    if p == '':
        return ''

    default_lib = os.path  # save default os.path
    if windows is not None:
        os.path = ntpath if windows else posixpath

    # Path('/blah').parts is ('/', 'blah')
    parts = split_path(p)

    drive = os.path.splitdrive(p)[0]
    if len(parts) > 1 and (
            parts[0] == drive
            or parts[0] == drive + '/'
            or parts[0] == drive + '\\'
            or parts[0] in ['\\', '/']):
        result = os.path.join(parts[0], parts[1])
    else:
        result = parts[0]

    os.path = default_lib  # restore default os.path
    return result


def safe_create_link(link_exists,link_new):
    """
    Create a symlink at link_new pointing to link_exists.
    
    If link_new already exists, make sure it's a link (not a file),
    and if it has a different target than link_exists, remove and re-create
    it.
    
    Errors if link_new already exists but it's not a link.
    """
    
    if os.path.exists(link_new) or os.path.islink(link_new):
        assert os.path.islink(link_new)
        if not os.readlink(link_new) == link_exists:
            os.remove(link_new)
            os.symlink(link_exists,link_new)
    else:
        os.symlink(link_exists,link_new)
        

def get_file_sizes(base_dir, convert_slashes=True):
    """
    Get sizes recursively for all files in base_dir, returning a dict mapping
    relative filenames to size.
    """
    
    relative_filenames = recursive_file_list(base_dir, convert_slashes=convert_slashes, 
                                             return_relative_paths=True)
    
    fn_to_size = {}
    for fn_relative in tqdm(relative_filenames):
        fn_abs = os.path.join(base_dir,fn_relative)
        fn_to_size[fn_relative] = os.path.getsize(fn_abs)
                   
    return fn_to_size
        

#%% Image-related path functions

def is_image_file(s: str, img_extensions: Container[str] = IMG_EXTENSIONS
                  ) -> bool:
    """
    Checks a file's extension against a hard-coded set of image file
    extensions.
    
    Does not check whether the file exists, only determines whether the filename
    implies it's an image file.
    """
    
    ext = os.path.splitext(s)[1]
    return ext.lower() in img_extensions


def find_image_strings(strings: Iterable[str]) -> List[str]:
    """
    Given a list of strings that are potentially image file names, looks for
    strings that actually look like image file names (based on extension).
    """
    
    return [s for s in strings if is_image_file(s)]


def find_images(dirname: str, recursive: bool = False, 
                return_relative_paths: bool = False, 
                convert_slashes: bool = False) -> List[str]:
    """
    Finds all files in a directory that look like image file names. Returns
    absolute paths unless return_relative_paths is set.  Uses the OS-native
    path separator unless convert_slashes is set, in which case will always
    use '/'.
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

def clean_filename(filename: str, allow_list: str = VALID_FILENAME_CHARS,
                   char_limit: int = CHAR_LIMIT, force_lower: bool = False) -> str:
    r"""
    Removes non-ASCII and other invalid filename characters (on any
    reasonable OS) from a filename, then trims to a maximum length.

    Does not allow :\/ by default, use clean_path if you want to preserve those.

    Adapted from
    https://gist.github.com/wassname/1393c4a57cfcbf03641dbc31886123b8
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


def clean_path(pathname: str, allow_list: str = VALID_PATH_CHARS,
               char_limit: int = CHAR_LIMIT, force_lower: bool = False) -> str:
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then trims to a maximum length.
    """
    
    return clean_filename(pathname, allow_list=allow_list, 
                          char_limit=char_limit, force_lower=force_lower)


def flatten_path(pathname: str, separator_chars: str = SEPARATOR_CHARS) -> str:
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then trims to a maximum length. Replaces all valid
    separators with '~'.
    """
    
    s = clean_path(pathname)
    for c in separator_chars:
        s = s.replace(c, '~')
    return s


#%% Platform-independent way to open files in their associated application

def environment_is_wsl():
    """
    Returns True if we're running in WSL
    """
    
    if sys.platform not in ('linux','posix'):
        return False
    platform_string = ' '.join(platform.uname()).lower()
    return 'microsoft' in platform_string and 'wsl' in platform_string
    

def wsl_path_to_windows_path(filename):
    """
    Converts a WSL path to a Windows path, or returns None if that's not possible.  E.g.
    converts:
        
    /mnt/e/a/b/c
    
    ...to:
        
    e:\a\b\c
    """
    
    result = subprocess.run(['wslpath', '-w', filename], text=True, capture_output=True)
    if result.returncode != 0:
        print('Could not convert path {} from WSL to Windows'.format(filename))
        return None
    return result.stdout.strip()
    
    
def open_file(filename, attempt_to_open_in_wsl_host=False, browser_name=None):
    """
    Opens [filename] in the default OS file handler for this file type.
    
    If attempt_to_open_in_wsl_host is True, and we're in WSL, attempts to open
    [filename] in the Windows host environment.
    
    If browser_name is not None, uses the webbrowser module to open the filename
    in the specified browser; see https://docs.python.org/3/library/webbrowser.html
    for supported browsers.  Falls back to the default file handler if webbrowser.open()
    fails.  In this case, attempt_to_open_in_wsl_host is ignored unless webbrowser.open() fails.
    
    If browser_name is 'default', use the system default.  This is different from the 
    parameter to webbrowser.get(), where None implies the system default.
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

def write_list_to_file(output_file: str, strings: Sequence[str]) -> None:
    """
    Writes a list of strings to either a JSON file or text file,
    depending on extension of the given file name.
    """
    
    with open(output_file, 'w') as f:
        if output_file.endswith('.json'):
            json.dump(strings, f, indent=1)
        else:
            f.write('\n'.join(strings))


def read_list_from_file(filename: str) -> List[str]:
    """
    Reads a json-formatted list of strings from a file.
    """
    
    assert filename.endswith('.json')
    with open(filename, 'r') as f:
        file_list = json.load(f)
    assert isinstance(file_list, list)
    for s in file_list:
        assert isinstance(s, str)
    return file_list


def _copy_file(input_output_tuple,overwrite=True,verbose=False):
    assert len(input_output_tuple) == 2
    source_fn = input_output_tuple[0]
    target_fn = input_output_tuple[1]
    if (not overwrite) and (os.path.isfile(target_fn)):
        if verbose:
            print('Skipping existing file {}'.format(target_fn))
        return
    shutil.copyfile(source_fn,target_fn)
    

def parallel_copy_files(input_file_to_output_file, max_workers=16, 
                        use_threads=True, overwrite=False, verbose=False):
    """
    Copy files from source to target according to the dict input_file_to_output_file.
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


#%% Zip functions

def zip_file(input_fn, output_fn=None, overwrite=False, verbose=False, compresslevel=9):
    """
    Zip a single file, by default writing to a new file called [input_fn].zip
    """
    
    basename = os.path.basename(input_fn)
    
    if output_fn is None:
        output_fn = input_fn + '.zip'
        
    if (not overwrite) and (os.path.isfile(output_fn)):
        print('Skipping existing file {}'.format(output_fn))
        return
    
    if verbose:
        print('Zipping {} to {} with level {}'.format(input_fn,output_fn,compresslevel))
    
    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_fn,arcname=basename,compresslevel=compresslevel,
                   compress_type=zipfile.ZIP_DEFLATED)

    return output_fn


def zip_files_into_single_zipfile(input_files, output_fn, arc_name_base,
                                  overwrite=False, verbose=False, compresslevel=9):
    """
    Zip all the files in [input_files] into [output_fn].  Archive names are relative to 
    arc_name_base.
    """
    
    if not overwrite:
        if os.path.isfile(output_fn):
            print('Zip file {} exists, skipping'.format(output_fn))
            return            
        
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
    Recursively zip everything in [input_folder] into a single zipfile, storing outputs as relative 
    paths.
    
    Defaults to writing to [input_folder].zip
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
    Zip one or more files to separate output files in parallel, leaving the 
    original files in place.  Each file is zipped to [filename].zip.
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
    Zip one or more folders to separate output files in parallel, leaving the 
    original folders in place.  Each folder is zipped to [folder_name].zip.
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
    Zip each file in [folder_name] to its own zipfile (filename.zip), optionally recursing.  To zip a whole
    folder into a single zipfile, use zip_folder().
    
    If required_token is not None, include only files that contain that token.
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
    Unzip a zipfile to the specified output folder, defaulting to the same location as
    the input file    
    """
    
    if output_folder is None:
        output_folder = os.path.dirname(input_file)
        
    with zipfile.ZipFile(input_file, 'r') as zf:
        zf.extractall(output_folder)
