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
import hashlib
import unicodedata
import zipfile
import tarfile
import webbrowser
import subprocess
import re

from zipfile import ZipFile
from datetime import datetime
from collections import defaultdict
from multiprocessing.pool import Pool, ThreadPool
from functools import partial
from shutil import which
from tqdm import tqdm

from megadetector.utils.ct_utils import is_iterable
from megadetector.utils.ct_utils import make_test_folder
from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.ct_utils import environment_is_wsl

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
    Enumerates files (not directories) in [base_dir].

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


def file_list(base_dir,
              convert_slashes=True,
              return_relative_paths=False,
              sort_files=True,
              recursive=False):
    """
    Trivial wrapper for recursive_file_list, which was a poor function name choice
    at the time, since I later wanted to add non-recursive lists, but it doesn't
    make sense to have a "recursive" option in a function called  "recursive_file_list".

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


def folder_list(base_dir,
                convert_slashes=True,
                return_relative_paths=False,
                sort_folders=True,
                recursive=False):
    """
    Enumerates folders (not files) in [base_dir].

    Args:
        base_dir (str): folder to enumerate
        convert_slashes (bool, optional): force forward slashes; if this is False, will use
            the native path separator
        return_relative_paths (bool, optional): return paths that are relative to [base_dir],
            rather than absolute paths
        sort_folders (bool, optional): force folders to be sorted, otherwise uses the sorting
            provided by os.walk()
        recursive (bool, optional): enumerate recursively

    Returns:
        list: list of folder names
    """

    assert os.path.isdir(base_dir), '{} is not a folder'.format(base_dir)

    folders = []

    if recursive:
        folders = []
        for root, dirs, _ in os.walk(base_dir):
            for d in dirs:
                folders.append(os.path.join(root, d))
    else:
        folders = os.listdir(base_dir)
        folders = [os.path.join(base_dir,fn) for fn in folders]
        folders = [fn for fn in folders if os.path.isdir(fn)]

    if return_relative_paths:
        folders = [os.path.relpath(fn,base_dir) for fn in folders]

    if convert_slashes:
        folders = [fn.replace('\\', '/') for fn in folders]

    if sort_folders:
        folders = sorted(folders)

    return folders


def folder_summary(folder,print_summary=True):
    """
    Returns (and optionally prints) a summary of [folder], including:

    * The total number of files
    * The total number of folders
    * The number of files for each extension

    Args:
        folder (str): folder to summarize
        print_summary (bool, optional): whether to print the summary

    Returns:
        dict: with fields "n_files", "n_folders", and "extension_to_count"
    """

    assert os.path.isdir(folder), '{} is not a folder'.format(folder)

    folders_relative = folder_list(folder,return_relative_paths=True,recursive=True)
    files_relative = file_list(folder,return_relative_paths=True,recursive=True)

    extension_to_count = defaultdict(int)

    for fn in files_relative:
        ext = os.path.splitext(fn)[1]
        extension_to_count[ext] += 1

    extension_to_count = sort_dictionary_by_value(extension_to_count,reverse=True)

    if print_summary:
        for extension in extension_to_count.keys():
            print('{}: {}'.format(extension,extension_to_count[extension]))
        print('')
        print('Total files: {}'.format(len(files_relative)))
        print('Total folders: {}'.format(len(folders_relative)))

    to_return = {}
    to_return['n_files'] = len(files_relative)
    to_return['n_folders'] = len(folders_relative)
    to_return['extension_to_count'] = extension_to_count

    return to_return


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

    # Edge cases
    if path == '':
        return ''
    if path is None:
        return None

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


def safe_create_link(link_exists,link_new):
    """
    Creates a symlink at [link_new] pointing to [link_exists].

    If [link_new] already exists, make sure it's a link (not a file),
    and if it has a different target than [link_exists], removes and re-creates
    it.

    Creates a *real* directory if necessary.

    Errors if [link_new] already exists but it's not a link.

    Args:
        link_exists (str): the source of the (possibly-new) symlink
        link_new (str): the target of the (possibly-new) symlink
    """

    # If the new file already exists...
    if os.path.exists(link_new) or os.path.islink(link_new):
        # Error if it's not already a link
        assert os.path.islink(link_new)
        # If it's already a link, and it points to the "exists" file,
        # leave it alone, otherwise redirect it.
        if not os.readlink(link_new) == link_exists:
            os.remove(link_new)
            os.symlink(link_exists,link_new)
    else:
        os.makedirs(os.path.dirname(link_new),exist_ok=True)
        os.symlink(link_exists,link_new)

 # ...def safe_create_link(...)


def remove_empty_folders(path, remove_root=False):
    """
    Recursively removes empty folders within the specified path.

    Args:
        path (str): the folder from which we should recursively remove
            empty folders.
        remove_root (bool, optional): whether to remove the root directory if
            it's empty after removing all empty subdirectories.  This will always
            be True during recursive calls.

    Returns:
        bool: True if the directory is empty after processing, False otherwise
    """

    # Verify that [path] is a directory
    if not os.path.isdir(path):
        return False

    # Track whether the current directory is empty
    is_empty = True

    # Iterate through all items in the directory
    for item in os.listdir(path):

        item_path = os.path.join(path, item)

        # If it's a directory, process it recursively
        if os.path.isdir(item_path):
            # If the subdirectory is empty after processing, it will be removed
            if not remove_empty_folders(item_path, True):
                # If the subdirectory is not empty, the current directory isn't empty either
                is_empty = False
        else:
            # If there's a file, the directory is not empty
            is_empty = False

    # If the directory is empty and we're supposed to remove it
    if is_empty and remove_root:
        try:
            os.rmdir(path)
        except Exception as e:
            print('Error removing directory {}: {}'.format(path,str(e)))
            is_empty = False

    return is_empty

# ...def remove_empty_folders(...)


def path_join(*paths, convert_slashes=True):
    r"""
    Wrapper for os.path.join that optionally converts backslashes to forward slashes.

    Args:
        *paths (variable-length set of strings): Path components to be joined.
        convert_slashes (bool, optional): whether to convert \\ to /

    Returns:
        A string with the joined path components.
    """

    joined_path = os.path.join(*paths)
    if convert_slashes:
        return joined_path.replace('\\', '/')
    else:
        return joined_path


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
                   force_lower=False,
                   remove_trailing_leading_whitespace=True):
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
        remove_trailing_leading_whitespace (bool, optional): remove trailing and
            leading whitespace from each component of a path, e.g. does not allow
            a/b/c /d.jpg
    Returns:
        str: cleaned version of [filename]
    """

    if remove_trailing_leading_whitespace:

        # Best effort to preserve the original separator
        separator = '/'
        if '\\' in filename:
            separator = '\\'

        filename = filename.replace('\\','/')
        components = filename.split('/')
        clean_components = [c.strip() for c in components]
        filename = separator.join(clean_components)
        if separator == '\\':
            filename = filename.replace('/','\\')

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
               force_lower=False,
               remove_trailing_leading_whitespace=True):
    """
    Removes non-ASCII and other invalid path characters (on any reasonable
    OS) from a path, then optionally trims to a maximum length.

    Args:
        pathname (str): path name to clean
        allow_list (str, optional): string containing all allowable filename characters
        char_limit (int, optional): maximum allowable filename length, if None will skip this
            step
        force_lower (bool, optional): convert the resulting filename to lowercase
        remove_trailing_leading_whitespace (bool, optional): remove trailing and
            leading whitespace from each component of a path, e.g. does not allow
            a/b/c /d.jpg

    Returns:
        str: cleaned version of [filename]
    """

    return clean_filename(pathname,
                          allow_list=allow_list,
                          char_limit=char_limit,
                          force_lower=force_lower,
                          remove_trailing_leading_whitespace=\
                            remove_trailing_leading_whitespace)


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


#%% WSL utilities

def wsl_path_to_windows_path(filename, failure_behavior='none'):
    r"""
    Converts a WSL path to a Windows path.  For example, converts:

    /mnt/e/a/b/c

    ...to:

    e:\a\b\c

    Args:
        filename (str): filename to convert
        failure_behavior (str, optional): what to do if the path can't be processed as a
            WSL path. 'none' to return None in this case, 'original' to return the original path.

    Returns:
        str: Windows equivalent to the WSL path [filename]
    """

    assert failure_behavior in ('none','original'), \
        'Unrecognized failure_behavior value {}'.format(failure_behavior)

    # Check whether the path follows the standard WSL mount pattern
    wsl_path_pattern = r'^/mnt/([a-zA-Z])(/.*)?$'
    match = re.match(wsl_path_pattern, filename)

    if match:

        # Extract the drive letter and the rest of the path
        drive_letter = match.group(1)
        path_remainder = match.group(2) if match.group(2) else ''

        # Convert forward slashes to backslashes for Windows
        path_remainder = path_remainder.replace('/', '\\')

        # Format the Windows path
        windows_path = f"{drive_letter}:{path_remainder}"
        return windows_path

    if failure_behavior == 'none':
        return None
    else:
        return filename

# ...def wsl_path_to_windows_path(...)


def windows_path_to_wsl_path(filename, failure_behavior='none'):
    r"""
    Converts a Windows path to a WSL path, or returns None if that's not possible.  E.g.
    converts:

    e:\a\b\c

    ...to:

    /mnt/e/a/b/c

    Args:
        filename (str): filename to convert
        failure_behavior (str, optional): what to do if the path can't be processed as a Windows path.
            'none' to return None in this case, 'original' to return the original path.

    Returns:
        str: WSL equivalent to the Windows path [filename]
    """

    assert failure_behavior in ('none','original'), \
        'Unrecognized failure_behavior value {}'.format(failure_behavior)

    filename = filename.replace('\\', '/')

    # Check whether the path follows a Windows drive letter pattern
    windows_path_pattern = r'^([a-zA-Z]):(/.*)?$'
    match = re.match(windows_path_pattern, filename)

    if match:
        # Extract the drive letter and the rest of the path
        drive_letter = match.group(1).lower()  # Convert to lowercase for WSL
        path_remainder = match.group(2) if match.group(2) else ''

        # Format the WSL path
        wsl_path = f"/mnt/{drive_letter}{path_remainder}"
        return wsl_path

    if failure_behavior == 'none':
        return None
    else:
        return filename

# ...def window_path_to_wsl_path(...)


#%% Platform-independent file openers

def open_file_in_chrome(filename):
    """
    Open a file in chrome, regardless of file type.  I typically use this to open
    .md files in Chrome.

    Args:
        filename (str): file to open

    Return:
        bool: whether the operation was successful
    """

    # Create URL
    abs_path = os.path.abspath(filename)

    system = platform.system()
    if system == 'Windows':
        url = f'file:///{abs_path.replace(os.sep, "/")}'
    else:  # macOS and Linux
        url = f'file://{abs_path}'

    # Determine the Chrome path
    if system == 'Windows':

        # This is a native Python module, but it only exists on Windows
        import winreg

        chrome_paths = [
            os.path.expanduser("~") + r"\AppData\Local\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ]

        # Default approach: run from a typical chrome location
        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.run([path, url])
                return True

        # Method 2: Check registry for Chrome path
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe") as key:
                chrome_path = winreg.QueryValue(key, None)
                if chrome_path and os.path.exists(chrome_path):
                    subprocess.run([chrome_path, url])
                    return True
        except Exception:
            pass

        # Method 3: Try alternate registry location
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                               r"Software\Google\Chrome\BLBeacon") as key:
                chrome_path = os.path.join(os.path.dirname(winreg.QueryValueEx(key, "version")[0]), "chrome.exe")
                if os.path.exists(chrome_path):
                    subprocess.run([chrome_path, url])
                    return True
        except Exception:
            pass

        # Method 4: Try system path or command
        for chrome_cmd in ["chrome", "chrome.exe", "googlechrome", "google-chrome"]:
            try:
                subprocess.run([chrome_cmd, url], shell=True)
                return True
            except Exception:
                continue

        # Method 5: Use Windows URL protocol handler
        try:
            os.startfile(url)
            return True
        except Exception:
            pass

        # Method 6: Use rundll32
        try:
            cmd = f'rundll32 url.dll,FileProtocolHandler {url}'
            subprocess.run(cmd, shell=True)
            return True
        except Exception:
            pass

    elif system == 'Darwin':

        chrome_paths = [
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
            os.path.expanduser('~/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        ]

        for path in chrome_paths:
            if os.path.exists(path):
                subprocess.run([path, url])
                return True

        # Fallback to 'open' command with Chrome as the app
        try:
            subprocess.run(['open', '-a', 'Google Chrome', url])
            return True
        except Exception:
            pass

    elif system == 'Linux':

        chrome_commands = ['google-chrome', 'chrome', 'chromium', 'chromium-browser']

        for cmd in chrome_commands:
            try:
                subprocess.run([cmd, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue

    print(f"Could not open {filename} in Chrome on {system}.")
    return False


def open_file(filename,
              attempt_to_open_in_wsl_host=False,
              browser_name=None):
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
        attempt_to_open_in_wsl_host (bool, optional): if this is True, and we're in WSL, attempts
            to open [filename] in the Windows host environment
        browser_name (str, optional): see above
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
            os.system("cmd.exe /C start {}".format(re.escape(windows_path)))

    else:

        opener = 'xdg-open'
        subprocess.call([opener, filename])

# ...def open_file(...)


#%% File list functions (as in, files that are lists of other filenames)

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


#%% File copying functions

def _copy_file(input_output_tuple,overwrite=True,verbose=False,move=False):
    """
    Internal function for copying files from within parallel_copy_files.
    """

    assert len(input_output_tuple) == 2
    source_fn = input_output_tuple[0]
    target_fn = input_output_tuple[1]
    if (not overwrite) and (os.path.isfile(target_fn)):
        if verbose:
            print('Skipping existing target file {}'.format(target_fn))
        return

    if move:
        action_string = 'Moving'
    else:
        action_string = 'Copying'

    if verbose:
        print('{} to {}'.format(action_string,target_fn))

    os.makedirs(os.path.dirname(target_fn),exist_ok=True)
    if move:
        shutil.move(source_fn, target_fn)
    else:
        shutil.copyfile(source_fn,target_fn)


def parallel_copy_files(input_file_to_output_file,
                        max_workers=16,
                        use_threads=True,
                        overwrite=False,
                        verbose=False,
                        move=False):
    """
    Copy (or move) files from source to target according to the dict input_file_to_output_file.

    Args:
        input_file_to_output_file (dict): dictionary mapping source files to the target files
            to which they should be copied
        max_workers (int, optional): number of concurrent workers; set to <=1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallel copying; ignored if max_workers <= 1
        overwrite (bool, optional): whether to overwrite existing destination files
        verbose (bool, optional): enable additional debug output
        move (bool, optional): move instead of copying
    """

    n_workers = min(max_workers,len(input_file_to_output_file))

    # Package the dictionary as a set of 2-tuples
    input_output_tuples = []
    for input_fn in input_file_to_output_file:
        input_output_tuples.append((input_fn,input_file_to_output_file[input_fn]))

    pool = None

    try:
        if use_threads:
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)

        with tqdm(total=len(input_output_tuples)) as pbar:
            for i,_ in enumerate(pool.imap_unordered(partial(_copy_file,
                                                            overwrite=overwrite,
                                                            verbose=verbose,
                                                            move=move),
                                                    input_output_tuples)):
                pbar.update()
    finally:
        pool.close()
        pool.join()
        if verbose:
            print("Pool closed and joined parallel file copying")

# ...def parallel_copy_files(...)


#%% File size functions

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
        verbose (bool, optional): enable additional debug output
        recursive (bool, optional): enumerate recursively, only relevant if [filenames] is a folder.
        convert_slashes (bool, optional): convert backslashes to forward slashes
        return_relative_paths (bool, optional): return relative paths; only relevant if [filenames]
            is a folder.

    Returns:
        dict: dictionary mapping filenames to file sizes in bytes
    """

    n_workers = min(max_workers,len(filenames))

    folder_name = None

    if isinstance(filenames,str):

        folder_name = filenames
        assert os.path.isdir(filenames), 'Could not find folder {}'.format(folder_name)

        if verbose:
            print('Enumerating files in {}'.format(folder_name))

        # Enumerate absolute paths here, we'll convert to relative later if requested
        filenames = recursive_file_list(folder_name,recursive=recursive,return_relative_paths=False)

    else:

        assert is_iterable(filenames), '[filenames] argument is neither a folder nor an iterable'

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

# ...def parallel_get_file_sizes(...)


#%% Compression (zip/tar) functions

def zip_file(input_fn, output_fn=None, overwrite=False, verbose=False, compress_level=9):
    """
    Zips a single file.

    Args:
        input_fn (str): file to zip
        output_fn (str, optional): target zipfile; if this is None, we'll use
            [input_fn].zip
        overwrite (bool, optional): whether to overwrite an existing target file
        verbose (bool, optional): enable existing debug console output
        compress_level (int, optional): compression level to use, between 0 and 9

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
        print('Zipping {} to {} with level {}'.format(input_fn,output_fn,compress_level))

    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_fn,
                   arcname=basename,
                   compresslevel=compress_level,
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


def zip_files_into_single_zipfile(input_files,
                                  output_fn,
                                  arc_name_base,
                                  overwrite=False,
                                  verbose=False,
                                  compress_level=9):
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
        compress_level (int, optional): compression level to use, between 0 and 9

    Returns:
        str: the output zipfile, whether we created it or determined that it already exists
    """

    if not overwrite:
        if os.path.isfile(output_fn):
            print('Zip file {} exists, skipping'.format(output_fn))
            return output_fn

    if verbose:
        print('Zipping {} files to {} (compression level {})'.format(
            len(input_files),output_fn,compress_level))

    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        for input_fn_abs in tqdm(input_files,disable=(not verbose)):
            input_fn_relative = os.path.relpath(input_fn_abs,arc_name_base)
            zipf.write(input_fn_abs,
                       arcname=input_fn_relative,
                       compresslevel=compress_level,
                       compress_type=zipfile.ZIP_DEFLATED)

    return output_fn


def zip_folder(input_folder, output_fn=None, overwrite=False, verbose=False, compress_level=9):
    """
    Recursively zip everything in [input_folder] into a single zipfile, storing files as paths
    relative to [input_folder].

    Args:
        input_folder (str): folder to zip
        output_fn (str, optional): output filename; if this is None, we'll write to [input_folder].zip
        overwrite (bool, optional): whether to overwrite an existing .tar file
        verbose (bool, optional): enable additional debug console output
        compress_level (int, optional): compression level to use, between 0 and 9

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
            input_folder,output_fn,compress_level))

    relative_filenames = recursive_file_list(input_folder,return_relative_paths=True)

    with ZipFile(output_fn,'w',zipfile.ZIP_DEFLATED) as zipf:
        for input_fn_relative in tqdm(relative_filenames,disable=(not verbose)):
            input_fn_abs = os.path.join(input_folder,input_fn_relative)
            zipf.write(input_fn_abs,
                       arcname=input_fn_relative,
                       compresslevel=compress_level,
                       compress_type=zipfile.ZIP_DEFLATED)

    return output_fn


def parallel_zip_files(input_files,
                       max_workers=16,
                       use_threads=True,
                       compress_level=9,
                       overwrite=False,
                       verbose=False):
    """
    Zips one or more files to separate output files in parallel, leaving the
    original files in place.  Each file is zipped to [filename].zip.

    Args:
        input_files (str): list of files to zip
        max_workers (int, optional): number of concurrent workers, set to <= 1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False); ignored if
            max_workers <= 1
        compress_level (int, optional): zip compression level between 0 and 9
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
          output_fn=None,overwrite=overwrite,verbose=verbose,compress_level=compress_level),
          input_files)):
            pbar.update()


def parallel_zip_folders(input_folders,
                         max_workers=16,
                         use_threads=True,
                         compress_level=9,
                         overwrite=False,
                         verbose=False):
    """
    Zips one or more folders to separate output files in parallel, leaving the
    original folders in place.  Each folder is zipped to [folder_name].zip.

    Args:
        input_folders (list): list of folders to zip
        max_workers (int, optional): number of concurrent workers, set to <= 1 to disable parallelism
        use_threads (bool, optional): whether to use threads (True) or processes (False); ignored if
            max_workers <= 1
        compress_level (int, optional): zip compression level between 0 and 9
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
                        compress_level=compress_level,verbose=verbose),
                input_folders)):
            pbar.update()


def zip_each_file_in_folder(folder_name,
                            recursive=False,
                            max_workers=16,
                            use_threads=True,
                            compress_level=9,
                            overwrite=False,
                            required_token=None,
                            verbose=False,
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
        compress_level (int, optional): zip compression level between 0 and 9
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
                       use_threads=use_threads,compress_level=compress_level,
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


#%% File hashing functions

def compute_file_hash(file_path, algorithm='sha256', allow_failures=True):
    """
    Compute the hash of a file.

    Adapted from:

    https://www.geeksforgeeks.org/python-program-to-find-hash-of-file/

    Args:
        file_path (str): the file to hash
        algorithm (str, optional): the hashing algorithm to use (e.g. md5, sha256)
        allow_failures (bool, optional): if True, read failures will silently return
            None; if false, read failures will raise exceptions

    Returns:
        str: the hash value for this file
    """

    try:

        hash_func = hashlib.new(algorithm)

        with open(file_path, 'rb') as file:
            while chunk := file.read(8192):  # Read the file in chunks of 8192 bytes
                hash_func.update(chunk)

        return str(hash_func.hexdigest())

    except Exception:

        if allow_failures:
            return None
        else:
            raise

# ...def compute_file_hash(...)


def parallel_compute_file_hashes(filenames,
                               max_workers=16,
                               use_threads=True,
                               recursive=True,
                               algorithm='sha256',
                               verbose=False):
    """
    Compute file hashes for a list or folder of images.

    Args:
        filenames (list or str): a list of filenames or a folder
        max_workers (int, optional): the number of parallel workers to use; set to <=1 to disable
            parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallelization
        algorithm (str, optional): the hashing algorithm to use (e.g. md5, sha256)
        recursive (bool, optional): if [filenames] is a folder, whether to enumerate recursively.
            Ignored if [filenames] is a list.
        verbose (bool, optional): enable additional debug output

    Returns:
        dict: a dict mapping filenames to hash values; values will be None for files that fail
        to load.
    """

    if isinstance(filenames,str) and os.path.isdir(filenames):
        if verbose:
            print('Enumerating files in {}'.format(filenames))
        filenames = recursive_file_list(filenames,recursive=recursive,return_relative_paths=False)

    n_workers = min(max_workers,len(filenames))

    if verbose:
        print('Computing hashes for {} files on {} workers'.format(len(filenames),n_workers))

    if n_workers <= 1:

        results = []
        for filename in filenames:
            results.append(compute_file_hash(filename,algorithm=algorithm,allow_failures=True))

    else:

        if use_threads:
            pool = ThreadPool(n_workers)
        else:
            pool = Pool(n_workers)

        results = list(tqdm(pool.imap(
            partial(compute_file_hash,algorithm=algorithm,allow_failures=True),
            filenames), total=len(filenames)))

    assert len(filenames) == len(results), 'Internal error in parallel_compute_file_hashes'

    to_return = {}
    for i_file,filename in enumerate(filenames):
        to_return[filename] = results[i_file]

    return to_return

# ...def parallel_compute_file_hashes(...)


#%% Tests

class TestPathUtils:
    """
    Tests for path_utils.py
    """

    def set_up(self):
        """
        Create a temporary directory for testing.
        """

        self.test_dir = make_test_folder(subfolder='megadetector/path_utils_tests')
        print('Using temporary folder {} for path utils testing'.format(self.test_dir))
        os.makedirs(self.test_dir, exist_ok=True)


    def tear_down(self):
        """
        Remove the temporary directory after tests.
        """

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)


    def test_is_image_file(self):
        """
        Test the is_image_file function.
        """

        assert is_image_file('test.jpg')
        assert is_image_file('test.jpeg')
        assert is_image_file('test.png')
        assert is_image_file('test.gif')
        assert is_image_file('test.bmp')
        assert is_image_file('test.tiff')
        assert is_image_file('test.TIF')
        assert not is_image_file('test.txt')
        assert not is_image_file('test.doc')
        assert is_image_file('path/to/image.JPG')
        assert not is_image_file('image')
        assert is_image_file('test.custom', img_extensions=['.custom'])
        assert not is_image_file('test.jpg', img_extensions=['.custom'])


    def test_find_image_strings(self):
        """
        Test the find_image_strings function.
        """

        strings = ['a.jpg', 'b.txt', 'c.PNG', 'd.gif', 'e.jpeg', 'f.doc']
        expected = ['a.jpg', 'c.PNG', 'd.gif', 'e.jpeg']
        assert sorted(find_image_strings(strings)) == sorted(expected)
        assert find_image_strings([]) == []
        assert find_image_strings(['no_image.txt', 'another.doc']) == []


    def test_find_images(self):
        """
        Test the find_images function.
        """

        # Create some dummy files
        img1_abs = os.path.join(self.test_dir, 'img1.jpg')
        img2_abs = os.path.join(self.test_dir, 'img2.PNG')
        txt1_abs = os.path.join(self.test_dir, 'text1.txt')
        open(img1_abs, 'w').close()
        open(img2_abs, 'w').close()
        open(txt1_abs, 'w').close()

        subdir = os.path.join(self.test_dir, 'subdir')
        os.makedirs(subdir, exist_ok=True)
        img3_abs = os.path.join(subdir, 'img3.jpeg')
        txt2_abs = os.path.join(subdir, 'text2.txt')
        open(img3_abs, 'w').close()
        open(txt2_abs, 'w').close()

        # Test non-recursive
        expected_non_recursive_abs = sorted([img1_abs.replace('\\', '/'), img2_abs.replace('\\', '/')])
        found_non_recursive_abs = find_images(self.test_dir, recursive=False, return_relative_paths=False)
        assert sorted(found_non_recursive_abs) == expected_non_recursive_abs

        # Test non-recursive, relative paths
        expected_non_recursive_rel = sorted(['img1.jpg', 'img2.PNG'])
        found_non_recursive_rel = find_images(self.test_dir, recursive=False, return_relative_paths=True)
        assert sorted(found_non_recursive_rel) == expected_non_recursive_rel

        # Test recursive
        expected_recursive_abs = sorted([
            img1_abs.replace('\\', '/'),
            img2_abs.replace('\\', '/'),
            img3_abs.replace('\\', '/')
        ])
        found_recursive_abs = find_images(self.test_dir, recursive=True, return_relative_paths=False)
        assert sorted(found_recursive_abs) == expected_recursive_abs

        # Test recursive, relative paths
        expected_recursive_rel = sorted([
            'img1.jpg',
            'img2.PNG',
            os.path.join('subdir', 'img3.jpeg').replace('\\', '/')
        ])
        found_recursive_rel = find_images(self.test_dir, recursive=True, return_relative_paths=True)
        assert sorted(found_recursive_rel) == expected_recursive_rel

        # Test with an empty directory
        empty_dir = os.path.join(self.test_dir, 'empty_dir')
        os.makedirs(empty_dir, exist_ok=True)
        assert find_images(empty_dir, recursive=True) == []

        # Test with a directory that doesn't exist (should assert)
        try:
            find_images(os.path.join(self.test_dir, 'non_existent_dir'))
            raise AssertionError("AssertionError not raised for non_existent_dir")
        except AssertionError:
            pass


    def test_recursive_file_list_and_file_list(self):
        """
        Test the recursive_file_list and file_list functions.
        """

        # Setup directory structure
        # test_dir/
        #   file1.txt
        #   file2.jpg
        #   subdir1/
        #     file3.txt
        #     subsubdir/
        #       file4.png
        #   subdir2/
        #     file5.doc

        list_dir = os.path.join(self.test_dir,'recursive_list')

        f1 = os.path.join(list_dir, 'file1.txt')
        f2 = os.path.join(list_dir, 'file2.jpg')
        subdir1 = os.path.join(list_dir, 'subdir1')
        os.makedirs(subdir1, exist_ok=True)
        f3 = os.path.join(subdir1, 'file3.txt')
        subsubdir = os.path.join(subdir1, 'subsubdir')
        os.makedirs(subsubdir, exist_ok=True)
        f4 = os.path.join(subsubdir, 'file4.png')
        subdir2 = os.path.join(list_dir, 'subdir2')
        os.makedirs(subdir2, exist_ok=True)
        f5 = os.path.join(subdir2, 'file5.doc')

        for filepath in [f1, f2, f3, f4, f5]:
            with open(filepath, 'w') as f:
                f.write('test')

        # Test recursive_file_list (recursive=True by default)
        expected_all_files_abs = sorted([
            f1.replace('\\', '/'), f2.replace('\\', '/'), f3.replace('\\', '/'),
            f4.replace('\\', '/'), f5.replace('\\', '/')
        ])
        all_files_abs = recursive_file_list(list_dir, convert_slashes=True,
                                            return_relative_paths=False)
        assert sorted(all_files_abs) == expected_all_files_abs

        # Test recursive_file_list with relative paths
        expected_all_files_rel = sorted([
            'file1.txt', 'file2.jpg',
            os.path.join('subdir1', 'file3.txt').replace('\\', '/'),
            os.path.join('subdir1', 'subsubdir', 'file4.png').replace('\\', '/'),
            os.path.join('subdir2', 'file5.doc').replace('\\', '/')
        ])
        all_files_rel = recursive_file_list(list_dir, convert_slashes=True,
                                            return_relative_paths=True)
        assert sorted(all_files_rel) == expected_all_files_rel

        # Test file_list (non-recursive by default via wrapper)
        expected_top_level_files_abs = sorted([f1.replace('\\', '/'), f2.replace('\\', '/')])
        top_level_files_abs = file_list(list_dir, convert_slashes=True,
                                        return_relative_paths=False, recursive=False)
        assert sorted(top_level_files_abs) == expected_top_level_files_abs

        # Test file_list (recursive explicitly) - should be same as recursive_file_list
        recursive_via_file_list = file_list(list_dir, convert_slashes=True,
                                            return_relative_paths=False, recursive=True)
        assert sorted(recursive_via_file_list) == expected_all_files_abs

        # Test with convert_slashes=False (use os.sep)
        #
        # Note: This test might be tricky if os.sep is '/', as no replacement happens. We'll check
        # that backslashes remain on Windows.
        if os.sep == '\\':
            f1_raw = os.path.join(list_dir, 'file1.txt')
            # Only one file for simplicity
            files_no_slash_conversion = file_list(list_dir, convert_slashes=False, recursive=False)
            assert any(f1_raw in s for s in files_no_slash_conversion)

        # Test with an empty directory
        empty_dir = os.path.join(list_dir, "empty_dir_for_files")
        os.makedirs(empty_dir, exist_ok=True)
        assert recursive_file_list(empty_dir) == []
        assert file_list(empty_dir, recursive=False) == []

        # Test with a non-existent directory
        try:
            recursive_file_list(os.path.join(list_dir, "non_existent_dir"))
            raise AssertionError("AssertionError not raised for non_existent_dir in recursive_file_list")
        except AssertionError:
            pass


    def test_folder_list(self):
        """
        Test the folder_list function.
        """

        # Setup directory structure
        # test_dir/
        #   subdir1/
        #     subsubdir1/
        #   subdir2/
        #   file1.txt (should be ignored)

        folder_list_dir = os.path.join(self.test_dir,'folder_list')

        subdir1 = os.path.join(folder_list_dir, 'subdir1')
        subsubdir1 = os.path.join(subdir1, 'subsubdir1')
        subdir2 = os.path.join(folder_list_dir, 'subdir2')
        os.makedirs(subdir1, exist_ok=True)
        os.makedirs(subsubdir1, exist_ok=True)
        os.makedirs(subdir2, exist_ok=True)
        with open(os.path.join(folder_list_dir, 'file1.txt'), 'w') as f:
            f.write('test')

        # Test non-recursive
        expected_folders_non_recursive_abs = sorted([
            subdir1.replace('\\', '/'), subdir2.replace('\\', '/')
        ])
        folders_non_recursive_abs = folder_list(folder_list_dir, recursive=False,
                                                return_relative_paths=False)
        assert sorted(folders_non_recursive_abs) == expected_folders_non_recursive_abs, \
            'Non-recursive folder list failured, expected:\n\n{}\n\nFound:\n\n{}'.format(
                str(expected_folders_non_recursive_abs),
                str(folders_non_recursive_abs)
            )

        # Test non-recursive, relative paths
        expected_folders_non_recursive_rel = sorted(['subdir1', 'subdir2'])
        folders_non_recursive_rel = folder_list(folder_list_dir, recursive=False,
                                                return_relative_paths=True)
        assert sorted(folders_non_recursive_rel) == expected_folders_non_recursive_rel

        # Test recursive
        expected_folders_recursive_abs = sorted([
            subdir1.replace('\\', '/'),
            subsubdir1.replace('\\', '/'),
            subdir2.replace('\\', '/')
        ])
        folders_recursive_abs = folder_list(folder_list_dir, recursive=True,
                                            return_relative_paths=False)
        assert sorted(folders_recursive_abs) == expected_folders_recursive_abs

        # Test recursive, relative paths
        expected_folders_recursive_rel = sorted([
            'subdir1',
            os.path.join('subdir1', 'subsubdir1').replace('\\', '/'),
            'subdir2'
        ])
        folders_recursive_rel = folder_list(folder_list_dir, recursive=True,
                                            return_relative_paths=True)
        assert sorted(folders_recursive_rel) == expected_folders_recursive_rel

        # Test with an empty directory (except for the file)
        empty_dir_for_folders = os.path.join(folder_list_dir, "empty_for_folders")
        os.makedirs(empty_dir_for_folders, exist_ok=True)
        with open(os.path.join(empty_dir_for_folders, 'temp.txt'), 'w') as f: f.write('t')
        assert folder_list(empty_dir_for_folders, recursive=True) == []
        assert folder_list(empty_dir_for_folders, recursive=False) == []

        # Test with a non-existent directory
        try:
            folder_list(os.path.join(self.test_dir, "non_existent_dir"))
            raise AssertionError("AssertionError not raised for non_existent_dir in folder_list")
        except AssertionError:
            pass


    def test_folder_summary(self):
        """
        Test the folder_summary function.
        """

        # test_dir/
        #   file1.txt
        #   img1.jpg
        #   subdir/
        #     file2.txt
        #     img2.png
        #     img3.png

        folder_summary_dir = os.path.join(self.test_dir,'folder_summary')

        f1 = os.path.join(folder_summary_dir, 'file1.txt')
        img1 = os.path.join(folder_summary_dir, 'img1.jpg')
        subdir = os.path.join(folder_summary_dir, 'subdir')
        os.makedirs(subdir, exist_ok=True)
        f2 = os.path.join(subdir, 'file2.txt')
        img2 = os.path.join(subdir, 'img2.png')
        img3 = os.path.join(subdir, 'img3.png')

        for filepath in [f1, img1, f2, img2, img3]:
            with open(filepath, 'w') as f:
                f.write('test')

        summary = folder_summary(folder_summary_dir, print_summary=False)

        assert summary['n_files'] == 5
        assert summary['n_folders'] == 1 # 'subdir'
        assert summary['extension_to_count']['.txt'] == 2
        assert summary['extension_to_count']['.jpg'] == 1
        assert summary['extension_to_count']['.png'] == 2

        # Check order (sorted by value, desc)
        #
        # The specific order of keys with the same counts can vary based on file system list
        # order.  We'll check that the counts are correct and the number of unique extensions is
        # right.
        assert len(summary['extension_to_count']) == 3


        empty_dir = os.path.join(folder_summary_dir, "empty_summary_dir")
        os.makedirs(empty_dir, exist_ok=True)
        empty_summary = folder_summary(empty_dir, print_summary=False)
        assert empty_summary['n_files'] == 0
        assert empty_summary['n_folders'] == 0
        assert empty_summary['extension_to_count'] == {}


    def test_fileparts(self):
        """
        Test the fileparts function.
        """

        assert fileparts('file') == ('', 'file', '')
        assert fileparts('file.txt') == ('', 'file', '.txt')
        assert fileparts(r'c:/dir/file.jpg') == ('c:/dir', 'file', '.jpg')
        assert fileparts('/dir/subdir/file.jpg') == ('/dir/subdir', 'file', '.jpg')
        assert fileparts(r'c:\dir\file') == (r'c:\dir', 'file', '')
        assert fileparts(r'c:\dir\file.tar.gz') == (r'c:\dir', 'file.tar', '.gz')
        assert fileparts('.bashrc') == ('', '.bashrc', '') # Hidden file, no extension
        assert fileparts('nodir/.bashrc') == ('nodir', '.bashrc', '')
        assert fileparts('a/b/c.d.e') == ('a/b', 'c.d', '.e')


    def test_insert_before_extension(self):
        """
        Test the insert_before_extension function.
        """

        assert insert_before_extension('file.ext', 'inserted') == 'file.inserted.ext'
        assert insert_before_extension('file', 'inserted') == 'file.inserted'
        assert insert_before_extension('path/to/file.ext', 'tag') == 'path/to/file.tag.ext'
        assert insert_before_extension('path/to/file', 'tag') == 'path/to/file.tag'
        assert insert_before_extension('file.tar.gz', 'new') == 'file.tar.new.gz'

        # Test with custom separator
        assert insert_before_extension('file.ext', 'inserted', separator='_') == 'file_inserted.ext'

        # Test with s=None (timestamp) - check format roughly
        fname_with_ts = insert_before_extension('file.ext', None)
        parts = fname_with_ts.split('.')
        # file.YYYY.MM.DD.HH.MM.SS.ext
        assert len(parts) >= 8 # file, Y, M, D, H, M, S, ext
        assert parts[0] == 'file'
        assert parts[-1] == 'ext'
        assert all(p.isdigit() for p in parts[1:-1])

        fname_no_ext_ts = insert_before_extension('file', '') # s is empty string, should also use timestamp
        parts_no_ext = fname_no_ext_ts.split('.')
        assert len(parts_no_ext) >= 7 # file, Y, M, D, H, M, S
        assert parts_no_ext[0] == 'file'
        assert all(p.isdigit() for p in parts_no_ext[1:])


    def test_split_path(self):
        """
        Test the split_path function.
        """

        if os.name == 'nt':
            assert split_path(r'c:\dir\subdir\file.txt') == ['c:\\', 'dir', 'subdir', 'file.txt']
            assert split_path('c:\\') == ['c:\\']
            # Test with mixed slashes, ntpath.split handles them
            assert split_path(r'c:/dir/subdir/file.txt') == ['c:/', 'dir', 'subdir', 'file.txt']
        else: # POSIX
            assert split_path('/dir/subdir/file.jpg') == ['/', 'dir', 'subdir', 'file.jpg']
            assert split_path('/') == ['/']

        assert split_path('dir/file.txt') == ['dir', 'file.txt']
        assert split_path('file.txt') == ['file.txt']
        assert split_path('') == ''
        assert split_path('.') == ['.']
        assert split_path('..') == ['..']
        assert split_path('../a/b') == ['..', 'a', 'b']


    def test_path_is_abs(self):
        """
        Test the path_is_abs function.
        """

        assert path_is_abs('/absolute/path')
        assert path_is_abs('c:/absolute/path')
        assert path_is_abs('C:\\absolute\\path')
        assert path_is_abs('\\\\server\\share\\path') # UNC path
        assert path_is_abs('c:file_without_slash_after_drive')

        assert not path_is_abs('relative/path')
        assert not path_is_abs('file.txt')
        assert not path_is_abs('../relative')
        assert not path_is_abs('')



    def test_safe_create_link_unix(self):
        """
        Test the safe_create_link function on Unix-like systems.
        """

        if os.name == 'nt':
            # print("Skipping test_safe_create_link_unix on Windows.")
            return

        source_file_path = os.path.join(self.test_dir, 'source.txt')
        link_path = os.path.join(self.test_dir, 'link.txt')
        other_source_path = os.path.join(self.test_dir, 'other_source.txt')

        with open(source_file_path, 'w') as f:
            f.write('source data')
        with open(other_source_path, 'w') as f:
            f.write('other data')

        # Create new link
        safe_create_link(source_file_path, link_path)
        assert os.path.islink(link_path)
        assert os.readlink(link_path) == source_file_path

        # Link already exists and points to the correct source
        safe_create_link(source_file_path, link_path) # Should do nothing
        assert os.path.islink(link_path)
        assert os.readlink(link_path) == source_file_path

        # Link already exists but points to a different source
        safe_create_link(other_source_path, link_path) # Should remove and re-create
        assert os.path.islink(link_path)
        assert os.readlink(link_path) == other_source_path

        # Link_new path exists and is a file (not a link)
        file_path_conflict = os.path.join(self.test_dir, 'conflict_file.txt')
        with open(file_path_conflict, 'w') as f:
            f.write('actual file')
        try:
            safe_create_link(source_file_path, file_path_conflict)
            raise AssertionError("AssertionError not raised for file conflict")
        except AssertionError:
            pass
        os.remove(file_path_conflict)

        # Link_new path exists and is a directory
        dir_path_conflict = os.path.join(self.test_dir, 'conflict_dir')
        os.makedirs(dir_path_conflict, exist_ok=True)
        try:
            safe_create_link(source_file_path, dir_path_conflict)
            raise AssertionError("AssertionError not raised for directory conflict")
        except AssertionError: # islink will be false
            pass
        shutil.rmtree(dir_path_conflict)


    def test_remove_empty_folders(self):
        """
        Test the remove_empty_folders function.
        """

        # test_dir/
        #   empty_top/
        #     empty_mid/
        #       empty_leaf/
        #   mixed_top/
        #     empty_mid_in_mixed/
        #       empty_leaf_in_mixed/
        #     non_empty_mid/
        #       file.txt
        #   non_empty_top/
        #     file_in_top.txt

        empty_top = os.path.join(self.test_dir, 'empty_top')
        empty_mid = os.path.join(empty_top, 'empty_mid')
        empty_leaf = os.path.join(empty_mid, 'empty_leaf')
        os.makedirs(empty_leaf, exist_ok=True)

        mixed_top = os.path.join(self.test_dir, 'mixed_top')
        empty_mid_in_mixed = os.path.join(mixed_top, 'empty_mid_in_mixed')
        empty_leaf_in_mixed = os.path.join(empty_mid_in_mixed, 'empty_leaf_in_mixed')
        os.makedirs(empty_leaf_in_mixed, exist_ok=True)
        non_empty_mid = os.path.join(mixed_top, 'non_empty_mid')
        os.makedirs(non_empty_mid, exist_ok=True)
        with open(os.path.join(non_empty_mid, 'file.txt'), 'w') as f:
            f.write('data')

        non_empty_top = os.path.join(self.test_dir, 'non_empty_top')
        os.makedirs(non_empty_top, exist_ok=True)
        with open(os.path.join(non_empty_top, 'file_in_top.txt'), 'w') as f:
            f.write('data')

        # Process empty_top - should remove all three
        remove_empty_folders(empty_top, remove_root=True)
        assert not os.path.exists(empty_top)
        assert not os.path.exists(empty_mid)
        assert not os.path.exists(empty_leaf)

        # Process mixed_top; should remove empty_leaf_in_mixed and empty_mid_in_mixed
        # but not mixed_top or non_empty_mid.
        remove_empty_folders(mixed_top, remove_root=True)
        assert os.path.exists(mixed_top) # mixed_top itself should remain
        assert not os.path.exists(empty_mid_in_mixed)
        assert not os.path.exists(empty_leaf_in_mixed)
        assert os.path.exists(non_empty_mid)
        assert os.path.exists(os.path.join(non_empty_mid, 'file.txt'))

        # Process non_empty_top; should remove nothing.
        remove_empty_folders(non_empty_top, remove_root=True)
        assert os.path.exists(non_empty_top)
        assert os.path.exists(os.path.join(non_empty_top, 'file_in_top.txt'))

        # Test with a file path (should do nothing and return False)
        file_path_for_removal = os.path.join(self.test_dir, 'a_file.txt')
        with open(file_path_for_removal, 'w') as f: f.write('t')
        assert not remove_empty_folders(file_path_for_removal, remove_root=True)
        assert os.path.exists(file_path_for_removal)

        # Test with remove_root=False for the top level
        another_empty_top = os.path.join(self.test_dir, 'another_empty_top')
        another_empty_mid = os.path.join(another_empty_top, 'another_empty_mid')
        os.makedirs(another_empty_mid)
        remove_empty_folders(another_empty_top, remove_root=False)
        assert os.path.exists(another_empty_top) # Root not removed
        assert not os.path.exists(another_empty_mid) # Mid removed


    def test_path_join(self):
        """
        Test the path_join function.
        """

        assert path_join('a', 'b', 'c') == 'a/b/c'
        assert path_join('a/b', 'c', 'd.txt') == 'a/b/c/d.txt'
        if os.name == 'nt':
            # On Windows, os.path.join uses '\', so convert_slashes=True should change it
            assert path_join('a', 'b', convert_slashes=True) == 'a/b'
            assert path_join('a', 'b', convert_slashes=False) == 'a\\b'
            assert path_join('c:\\', 'foo', 'bar', convert_slashes=True) == 'c:/foo/bar'
            assert path_join('c:\\', 'foo', 'bar', convert_slashes=False) == 'c:\\foo\\bar'
        else:
            # On POSIX, os.path.join uses '/', so convert_slashes=False should still be '/'
            assert path_join('a', 'b', convert_slashes=False) == 'a/b'

        assert path_join('a', '', 'b') == 'a/b' # os.path.join behavior
        assert path_join('/a', 'b') == '/a/b'
        assert path_join('a', '/b') == '/b' # '/b' is absolute


    def test_filename_cleaning(self):
        """
        Test clean_filename, clean_path, and flatten_path functions.
        """

        # clean_filename
        assert clean_filename("test file.txt") == "test file.txt"
        assert clean_filename("test*file?.txt", char_limit=10) == "testfile.t"
        assert clean_filename("TestFile.TXT", force_lower=True) == "testfile.txt"
        assert clean_filename("file:with<illegal>chars.txt") == "filewithillegalchars.txt"

        s = " accented_name_éà.txt"

        assert clean_filename(s,
                              remove_trailing_leading_whitespace=False) == " accented_name_ea.txt", \
            'clean_filename with remove_trailing_leading_whitespace=False: {}'.format(
                clean_filename(s, remove_trailing_leading_whitespace=False))

        assert clean_filename(s, remove_trailing_leading_whitespace=True) == "accented_name_ea.txt", \
            'clean_filename with remove_trailing_leading_whitespace=False: {}'.format(
                clean_filename(s, remove_trailing_leading_whitespace=True))

        # Separators are not allowed by default in clean_filename
        assert clean_filename("path/to/file.txt") == "pathtofile.txt"

        # clean_path
        assert clean_path("path/to/file.txt") == "path/to/file.txt" # slashes allowed
        assert clean_path("path\\to\\file.txt") == "path\\to\\file.txt" # backslashes allowed
        assert clean_path("path:to:file.txt") == "path:to:file.txt" # colons allowed
        assert clean_path("path/to<illegal>/file.txt") == "path/toillegal/file.txt"

        # flatten_path
        assert flatten_path("path/to/file.txt") == "path~to~file.txt"
        assert flatten_path("path:to:file.txt", separator_char_replacement='_') == "path_to_file.txt"
        assert flatten_path("path\\to/file:name.txt") == "path~to~file~name.txt"
        assert flatten_path("path/to<illegal>/file.txt") == "path~toillegal~file.txt"


    def test_is_executable(self):
        """
        Test the is_executable function.
        This is a basic test; comprehensive testing is environment-dependent.
        """

        # Hard to test reliably across all systems without knowing what's on PATH.
        if os.name == 'nt':
            assert is_executable('cmd.exe')
            assert not is_executable('non_existent_executable_blah_blah')
        else:
            assert is_executable('ls')
            assert is_executable('sh')
            assert not is_executable('non_existent_executable_blah_blah')


    def test_write_read_list_to_file(self):
        """
        Test write_list_to_file and read_list_from_file functions.
        """

        test_list = ["item1", "item2 with space", "item3/with/slash"]

        # Test with .json
        json_file_path = os.path.join(self.test_dir, "test_list.json")
        write_list_to_file(json_file_path, test_list)
        read_list_json = read_list_from_file(json_file_path)
        assert test_list == read_list_json

        # Test with .txt
        txt_file_path = os.path.join(self.test_dir, "test_list.txt")
        write_list_to_file(txt_file_path, test_list)
        # read_list_from_file is specifically for JSON, so we read .txt manually
        with open(txt_file_path, 'r') as f:
            read_list_txt = [line.strip() for line in f.readlines()]
        assert test_list == read_list_txt

        # Test reading non-existent json
        try:
            read_list_from_file(os.path.join(self.test_dir,"non_existent.json"))
            raise AssertionError("FileNotFoundError not raised")
        except FileNotFoundError:
            pass

        # Test reading a non-json file with read_list_from_file (should fail parsing)
        non_json_path = os.path.join(self.test_dir, "not_a_list.json")
        with open(non_json_path, 'w') as f: f.write("this is not json")
        try:
             read_list_from_file(non_json_path)
             raise AssertionError("json.JSONDecodeError not raised")
        except json.JSONDecodeError:
             pass


    def test_parallel_copy_files(self):
        """
        Test the parallel_copy_files function (with max_workers=1 for test simplicity).
        """

        source_dir = os.path.join(self.test_dir, "copy_source")
        target_dir = os.path.join(self.test_dir, "copy_target")
        os.makedirs(source_dir, exist_ok=True)

        file_mappings = {}
        source_files_content = {}

        for i in range(3):
            src_fn = f"file{i}.txt"
            src_path = os.path.join(source_dir, src_fn)
            if i == 0:
                tgt_fn = f"copied_file{i}.txt"
                tgt_path = os.path.join(target_dir, tgt_fn)
            else:
                tgt_fn = f"copied_file{i}_subdir.txt"
                tgt_path = os.path.join(target_dir, f"sub{i}", tgt_fn)

            content = f"content of file {i}"
            with open(src_path, 'w') as f:
                f.write(content)

            file_mappings[src_path] = tgt_path
            source_files_content[tgt_path] = content

        # Test copy
        parallel_copy_files(file_mappings, max_workers=1, use_threads=True, overwrite=False)
        for tgt_path, expected_content in source_files_content.items():
            assert os.path.exists(tgt_path)
            with open(tgt_path, 'r') as f:
                assert f.read() == expected_content

        existing_target_path = list(source_files_content.keys())[0]
        with open(existing_target_path, 'w') as f:
            f.write("old content")

        parallel_copy_files(file_mappings, max_workers=1, use_threads=True, overwrite=False)
        with open(existing_target_path, 'r') as f:
            assert f.read() == "old content"

        parallel_copy_files(file_mappings, max_workers=1, use_threads=True, overwrite=True)
        with open(existing_target_path, 'r') as f:
            assert f.read() == source_files_content[existing_target_path]

        for src_path_orig, tgt_path_orig in file_mappings.items(): # Re-create source for move
            with open(src_path_orig, 'w') as f:
                f.write(source_files_content[tgt_path_orig])

        parallel_copy_files(file_mappings, max_workers=1, use_threads=True, move=True, overwrite=True)
        for src_path, tgt_path in file_mappings.items():
            assert not os.path.exists(src_path)
            assert os.path.exists(tgt_path)
            with open(tgt_path, 'r') as f:
                assert f.read() == source_files_content[tgt_path]


    def test_get_file_sizes(self):
        """
        Test get_file_sizes and parallel_get_file_sizes functions.
        """

        file_sizes_test_dir = os.path.join(self.test_dir,'file_sizes')
        os.makedirs(file_sizes_test_dir,exist_ok=True)

        f1_path = os.path.join(file_sizes_test_dir, 'file1.txt')
        content1 = "0123456789" # 10 bytes
        with open(f1_path, 'w') as f:
            f.write(content1)

        subdir_path = os.path.join(file_sizes_test_dir, 'subdir')
        os.makedirs(subdir_path, exist_ok=True)
        f2_path = os.path.join(subdir_path, 'file2.txt')
        content2 = "01234567890123456789" # 20 bytes
        with open(f2_path, 'w') as f:
            f.write(content2)

        sizes_relative = get_file_sizes(file_sizes_test_dir)
        expected_sizes_relative = {
            'file1.txt': len(content1),
            os.path.join('subdir', 'file2.txt').replace('\\','/'): len(content2)
        }
        assert sizes_relative == expected_sizes_relative

        file_list_abs = [f1_path, f2_path]
        sizes_parallel_abs = parallel_get_file_sizes(file_list_abs, max_workers=1)
        expected_sizes_parallel_abs = {
            f1_path.replace('\\','/'): len(content1),
            f2_path.replace('\\','/'): len(content2)
        }
        assert sizes_parallel_abs == expected_sizes_parallel_abs

        sizes_parallel_folder_abs = parallel_get_file_sizes(file_sizes_test_dir,
                                                            max_workers=1,
                                                            return_relative_paths=False)
        assert sizes_parallel_folder_abs == expected_sizes_parallel_abs

        sizes_parallel_folder_rel = parallel_get_file_sizes(file_sizes_test_dir,
                                                            max_workers=1,
                                                            return_relative_paths=True)
        assert sizes_parallel_folder_rel == expected_sizes_relative

        non_existent_file = os.path.join(file_sizes_test_dir, "no_such_file.txt")
        sizes_with_error = parallel_get_file_sizes([f1_path, non_existent_file],
                                                   max_workers=1)
        expected_with_error = {
            f1_path.replace('\\','/'): len(content1),
            non_existent_file.replace('\\','/'): None
        }
        assert sizes_with_error == expected_with_error


    def test_zip_file_and_unzip_file(self):
        """
        Test zip_file and unzip_file functions.
        """

        file_to_zip_name = "test_zip_me.txt"
        file_to_zip_path = os.path.join(self.test_dir, file_to_zip_name)
        content = "This is the content to be zipped."
        with open(file_to_zip_path, 'w') as f:
            f.write(content)

        default_zip_output_path = file_to_zip_path + ".zip"
        returned_zip_path = zip_file(file_to_zip_path)
        assert returned_zip_path == default_zip_output_path
        assert os.path.exists(default_zip_output_path)

        unzip_dir_default = os.path.join(self.test_dir, "unzip_default")
        os.makedirs(unzip_dir_default, exist_ok=True)
        unzip_file(default_zip_output_path, unzip_dir_default)
        unzipped_file_path_default = os.path.join(unzip_dir_default, file_to_zip_name)
        assert os.path.exists(unzipped_file_path_default)
        with open(unzipped_file_path_default, 'r') as f:
            assert f.read() == content

        custom_zip_output_name = "custom_archive.zip"
        custom_zip_output_path = os.path.join(self.test_dir, custom_zip_output_name)
        zip_file(file_to_zip_path, output_fn=custom_zip_output_path, overwrite=True)
        assert os.path.exists(custom_zip_output_path)

        zip_in_subdir_path = os.path.join(self.test_dir, "subdir_zip", "my.zip")
        file_in_subdir_name = "file_for_subdir_zip.txt"
        file_in_subdir_path = os.path.join(self.test_dir,"subdir_zip", file_in_subdir_name)
        os.makedirs(os.path.dirname(zip_in_subdir_path), exist_ok=True)
        with open(file_in_subdir_path, "w") as f: f.write("sub dir content")
        zip_file(file_in_subdir_path, output_fn=zip_in_subdir_path)

        unzip_file(zip_in_subdir_path, output_folder=None)
        unzipped_in_same_dir_path = os.path.join(os.path.dirname(zip_in_subdir_path), file_in_subdir_name)
        assert os.path.exists(unzipped_in_same_dir_path)
        with open(unzipped_in_same_dir_path, 'r') as f:
            assert f.read() == "sub dir content"


    def test_zip_folder(self):
        """
        Test the zip_folder function.
        """

        folder_to_zip = os.path.join(self.test_dir, "folder_to_zip")
        os.makedirs(folder_to_zip, exist_ok=True)

        file1_name = "file1.txt"; path1 = os.path.join(folder_to_zip, file1_name)
        file2_name = "file2.log"; path2 = os.path.join(folder_to_zip, file2_name)
        subdir_name = "sub"; subdir_path = os.path.join(folder_to_zip, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        file3_name = "file3.dat"; path3 = os.path.join(subdir_path, file3_name)

        content1 = "content1"; content2 = "content2"; content3 = "content3"
        with open(path1, 'w') as f: f.write(content1)
        with open(path2, 'w') as f: f.write(content2)
        with open(path3, 'w') as f: f.write(content3)

        default_zip_path = folder_to_zip + ".zip"
        zip_folder(folder_to_zip, output_fn=None, overwrite=True)
        assert os.path.exists(default_zip_path)

        unzip_output_dir = os.path.join(self.test_dir, "unzipped_folder_content")
        os.makedirs(unzip_output_dir, exist_ok=True)
        unzip_file(default_zip_path, unzip_output_dir)

        assert os.path.exists(os.path.join(unzip_output_dir, file1_name))
        assert os.path.exists(os.path.join(unzip_output_dir, file2_name))
        assert os.path.exists(os.path.join(unzip_output_dir, subdir_name, file3_name))
        with open(os.path.join(unzip_output_dir, file1_name), 'r')as f: assert f.read() == content1
        with open(os.path.join(unzip_output_dir, file2_name), 'r')as f: assert f.read() == content2
        with open(os.path.join(unzip_output_dir, subdir_name, file3_name), 'r')as f: assert f.read() == content3

        mtime_before = os.path.getmtime(default_zip_path)
        zip_folder(folder_to_zip, output_fn=None, overwrite=False)
        mtime_after = os.path.getmtime(default_zip_path)
        assert mtime_before == mtime_after


    def test_zip_files_into_single_zipfile(self):
        """
        Test zip_files_into_single_zipfile.
        """

        file1_path = os.path.join(self.test_dir, "zfs_file1.txt")
        content1 = "content for zfs1"
        with open(file1_path, 'w') as f: f.write(content1)

        subdir_for_zfs = os.path.join(self.test_dir, "zfs_subdir")
        os.makedirs(subdir_for_zfs, exist_ok=True)
        file2_path = os.path.join(subdir_for_zfs, "zfs_file2.log")
        content2 = "content for zfs2"
        with open(file2_path, 'w') as f: f.write(content2)

        input_files = [file1_path, file2_path]
        output_zip_path = os.path.join(self.test_dir, "multi_file_archive.zip")
        zip_files_into_single_zipfile(input_files, output_zip_path, arc_name_base=self.test_dir, overwrite=True)
        assert os.path.exists(output_zip_path)

        unzip_dir = os.path.join(self.test_dir, "unzip_multi_file")
        os.makedirs(unzip_dir, exist_ok=True)
        unzip_file(output_zip_path, unzip_dir)

        expected_unzipped_file1 = os.path.join(unzip_dir, os.path.relpath(file1_path, self.test_dir))
        expected_unzipped_file2 = os.path.join(unzip_dir, os.path.relpath(file2_path, self.test_dir))

        assert os.path.exists(expected_unzipped_file1)
        with open(expected_unzipped_file1, 'r') as f: assert f.read() == content1
        assert os.path.exists(expected_unzipped_file2)
        assert os.path.basename(expected_unzipped_file2) == "zfs_file2.log"
        assert os.path.basename(os.path.dirname(expected_unzipped_file2)) == "zfs_subdir"
        with open(expected_unzipped_file2, 'r') as f: assert f.read() == content2


    def test_add_files_to_single_tar_file(self):
        """
        Test add_files_to_single_tar_file.
        """

        file1_path = os.path.join(self.test_dir, "tar_file1.txt")
        content1 = "content for tar1"
        with open(file1_path, 'w') as f: f.write(content1)

        subdir_for_tar = os.path.join(self.test_dir, "tar_subdir")
        os.makedirs(subdir_for_tar, exist_ok=True)
        file2_path = os.path.join(subdir_for_tar, "tar_file2.log")
        content2 = "content for tar2"
        with open(file2_path, 'w') as f: f.write(content2)

        input_files = [file1_path, file2_path]
        output_tar_path = os.path.join(self.test_dir, "archive.tar.gz")

        add_files_to_single_tar_file(input_files, output_tar_path, arc_name_base=self.test_dir,
                                     overwrite=True, mode='x:gz')
        assert os.path.exists(output_tar_path)

        un_tar_dir = os.path.join(self.test_dir, "un_tar_contents")
        os.makedirs(un_tar_dir, exist_ok=True)
        with tarfile.open(output_tar_path, 'r:gz') as tf:
            # The "filter" option was added as of Python 3.12, and *not* specifying
            # filter=None will change behavior as of Python 3.14.  We want the unmodified
            # behavior, but we want to support Python <3.12, so we do a version check.
            if sys.version_info >= (3, 12):
                tf.extractall(path=un_tar_dir, filter=None)
            else:
                tf.extractall(path=un_tar_dir)

        expected_untarred_file1 = os.path.join(un_tar_dir, os.path.relpath(file1_path, self.test_dir))
        expected_untarred_file2 = os.path.join(un_tar_dir, os.path.relpath(file2_path, self.test_dir))

        assert os.path.exists(expected_untarred_file1)
        with open(expected_untarred_file1, 'r') as f: assert f.read() == content1
        assert os.path.exists(expected_untarred_file2)
        with open(expected_untarred_file2, 'r') as f: assert f.read() == content2


    def test_parallel_zip_individual_files_and_folders(self):
        """
        Test parallel_zip_files, parallel_zip_folders, and zip_each_file_in_folder.
        """

        file1_to_zip = os.path.join(self.test_dir, "pz_file1.txt")
        file2_to_zip = os.path.join(self.test_dir, "pz_file2.txt")
        with open(file1_to_zip, 'w') as f: f.write("pz_content1")
        with open(file2_to_zip, 'w') as f: f.write("pz_content2")

        parallel_zip_files([file1_to_zip, file2_to_zip], max_workers=1, overwrite=True)
        assert os.path.exists(file1_to_zip + ".zip")
        assert os.path.exists(file2_to_zip + ".zip")
        unzip_dir_pz = os.path.join(self.test_dir, "unzip_pz")
        unzip_file(file1_to_zip + ".zip", unzip_dir_pz)
        assert os.path.exists(os.path.join(unzip_dir_pz, os.path.basename(file1_to_zip)))

        folder1_to_zip = os.path.join(self.test_dir, "pz_folder1")
        os.makedirs(folder1_to_zip, exist_ok=True)
        with open(os.path.join(folder1_to_zip, "pf1.txt"), 'w') as f: f.write("pf1_content")
        folder2_to_zip = os.path.join(self.test_dir, "pz_folder2")
        os.makedirs(folder2_to_zip, exist_ok=True)
        with open(os.path.join(folder2_to_zip, "pf2.txt"), 'w') as f: f.write("pf2_content")

        parallel_zip_folders([folder1_to_zip, folder2_to_zip], max_workers=1, overwrite=True)
        assert os.path.exists(folder1_to_zip + ".zip")
        assert os.path.exists(folder2_to_zip + ".zip")
        unzip_dir_pzf = os.path.join(self.test_dir, "unzip_pzf")
        unzip_file(folder1_to_zip + ".zip", unzip_dir_pzf)
        assert os.path.exists(os.path.join(unzip_dir_pzf, "pf1.txt"))

        zef_folder = os.path.join(self.test_dir, "zef_test_folder")
        os.makedirs(zef_folder, exist_ok=True)
        zef_file1 = os.path.join(zef_folder, "zef1.txt")
        zef_file2_png = os.path.join(zef_folder, "zef2.png")
        zef_file3_zip = os.path.join(zef_folder, "zef3.zip")
        zef_subdir = os.path.join(zef_folder, "zef_sub")
        os.makedirs(zef_subdir, exist_ok=True)
        zef_file_in_sub = os.path.join(zef_subdir, "zef_subfile.txt")

        for p_path in [zef_file1, zef_file2_png, zef_file3_zip, zef_file_in_sub]:
            with open(p_path, 'w') as f: f.write(f"content of {os.path.basename(p_path)}")

        zip_each_file_in_folder(zef_folder, recursive=False, max_workers=1, overwrite=True)
        assert os.path.exists(zef_file1 + ".zip")
        assert os.path.exists(zef_file2_png + ".zip")
        assert not os.path.exists(zef_file3_zip + ".zip")
        assert not os.path.exists(zef_file_in_sub + ".zip")

        if os.path.exists(zef_file1 + ".zip"): os.remove(zef_file1 + ".zip")
        if os.path.exists(zef_file2_png + ".zip"): os.remove(zef_file2_png + ".zip")

        zip_each_file_in_folder(zef_folder, recursive=True, max_workers=1, overwrite=True)
        assert os.path.exists(zef_file1 + ".zip")
        assert os.path.exists(zef_file2_png + ".zip")
        assert not os.path.exists(zef_file3_zip + ".zip")
        assert os.path.exists(zef_file_in_sub + ".zip")

        if os.path.exists(zef_file1 + ".zip"): os.remove(zef_file1 + ".zip")
        if os.path.exists(zef_file2_png + ".zip"): os.remove(zef_file2_png + ".zip")
        if os.path.exists(zef_file_in_sub + ".zip"): os.remove(zef_file_in_sub + ".zip")
        zip_each_file_in_folder(zef_folder, recursive=True, required_token="zef1", max_workers=1, overwrite=True)
        assert os.path.exists(zef_file1 + ".zip")
        assert not os.path.exists(zef_file2_png + ".zip")
        assert not os.path.exists(zef_file_in_sub + ".zip")

        if os.path.exists(zef_file1 + ".zip"): os.remove(zef_file1 + ".zip")
        dummy_to_zip = os.path.join(zef_folder,"dummy.txt")
        with open(dummy_to_zip,'w') as f: f.write('d')
        zip_each_file_in_folder(zef_folder, recursive=False, exclude_zip=False, max_workers=1, overwrite=True)
        assert os.path.exists(dummy_to_zip + ".zip")
        assert os.path.exists(zef_file3_zip + ".zip")
        if os.path.exists(dummy_to_zip + ".zip"): os.remove(dummy_to_zip + ".zip")
        if os.path.exists(zef_file3_zip + ".zip"): os.remove(zef_file3_zip + ".zip")


    def test_compute_file_hash(self):
        """
        Test compute_file_hash and parallel_compute_file_hashes.
        """

        file1_name = "hash_me1.txt"
        file1_path = os.path.join(self.test_dir, file1_name)
        content1 = "This is a test string for hashing."
        with open(file1_path, 'w') as f:
            f.write(content1)

        file2_name = "hash_me2.txt"
        file2_path = os.path.join(self.test_dir, file2_name)
        with open(file2_path, 'w') as f:
            f.write(content1)

        file3_name = "hash_me3.txt"
        file3_path = os.path.join(self.test_dir, file3_name)
        content3 = "This is a different test string for hashing."
        with open(file3_path, 'w') as f:
            f.write(content3)

        expected_hash_content1_sha256 = \
            "c56f19d76df6a09e49fe0d9ce7b1bc7f1dbd582f668742bede65c54c47d5bcf4".lower()
        expected_hash_content3_sha256 = \
            "23013ff7e93264317f7b2fc0e9a217649f2dc0b11ca7e0bd49632424b70b6680".lower()

        hash1 = compute_file_hash(file1_path)
        hash2 = compute_file_hash(file2_path)
        hash3 = compute_file_hash(file3_path)
        assert hash1 == expected_hash_content1_sha256
        assert hash2 == expected_hash_content1_sha256
        assert hash1 != hash3
        assert hash3 == expected_hash_content3_sha256

        expected_hash_content1_md5 = "94b971f1f8cdb23c2af82af73160d4b0".lower()
        hash1_md5 = compute_file_hash(file1_path, algorithm='md5')
        assert hash1_md5 == expected_hash_content1_md5

        non_existent_path = os.path.join(self.test_dir, "no_such_file.txt")
        assert compute_file_hash(non_existent_path, allow_failures=True) is None
        try:
             compute_file_hash(non_existent_path, allow_failures=False)
             raise AssertionError("FileNotFoundError not raised for compute_file_hash")
        except FileNotFoundError:
             pass

        files_to_hash = [file1_path, file3_path, non_existent_path]
        hashes_parallel = parallel_compute_file_hashes(files_to_hash, max_workers=1)

        norm_f1 = file1_path.replace('\\','/')
        norm_f3 = file3_path.replace('\\','/')
        norm_non = non_existent_path.replace('\\','/')

        expected_parallel_hashes = {
            norm_f1: expected_hash_content1_sha256,
            norm_f3: expected_hash_content3_sha256,
            norm_non: None
        }
        hashes_parallel_norm = {k.replace('\\','/'): v for k,v in hashes_parallel.items()}
        assert hashes_parallel_norm == expected_parallel_hashes

        hash_folder = os.path.join(self.test_dir, "hash_test_folder")
        os.makedirs(hash_folder, exist_ok=True)
        h_f1_name = "h_f1.txt"; h_f1_path = os.path.join(hash_folder, h_f1_name)
        h_f2_name = "h_f2.txt"; h_f2_path = os.path.join(hash_folder, h_f2_name)
        with open(h_f1_path, 'w') as f: f.write(content1)
        with open(h_f2_path, 'w') as f: f.write(content3)

        hashes_folder_parallel = parallel_compute_file_hashes(hash_folder, recursive=False, max_workers=1)
        norm_hf1 = h_f1_path.replace('\\','/')
        norm_hf2 = h_f2_path.replace('\\','/')
        expected_folder_hashes = {
            norm_hf1: expected_hash_content1_sha256,
            norm_hf2: expected_hash_content3_sha256
        }
        hashes_folder_parallel_norm = {k.replace('\\','/'): v for k,v in hashes_folder_parallel.items()}
        assert hashes_folder_parallel_norm == expected_folder_hashes


def test_path_utils():
    """
    Runs all tests in the TestPathUtils class.
    """

    test_instance = TestPathUtils()
    test_instance.set_up()

    try:

        test_instance.test_is_image_file()
        test_instance.test_find_image_strings()
        test_instance.test_find_images()
        test_instance.test_recursive_file_list_and_file_list()
        test_instance.test_folder_list()
        test_instance.test_folder_summary()
        test_instance.test_fileparts()
        test_instance.test_insert_before_extension()
        test_instance.test_split_path()
        test_instance.test_path_is_abs()
        test_instance.test_safe_create_link_unix()
        test_instance.test_remove_empty_folders()
        test_instance.test_path_join()
        test_instance.test_filename_cleaning()
        test_instance.test_is_executable()
        test_instance.test_write_read_list_to_file()
        test_instance.test_parallel_copy_files()
        test_instance.test_get_file_sizes()
        test_instance.test_zip_file_and_unzip_file()
        test_instance.test_zip_folder()
        test_instance.test_zip_files_into_single_zipfile()
        test_instance.test_add_files_to_single_tar_file()
        test_instance.test_parallel_zip_individual_files_and_folders()
        test_instance.test_compute_file_hash()

    finally:

        test_instance.tear_down()
