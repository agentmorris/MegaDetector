"""

directory_listing.py

Script for creating Apache-style HTML directory listings for a local directory
and all its subdirectories.

Also includes a preview of a jpg file (the first in an alphabetical list),
if present.

"""

#%% Imports

import os
import sys
import argparse

from megadetector.utils.path_utils import is_image_file


#%% Directory enumeration functions

def _create_plain_index(root, dirs, files, dirname=None):
    """
    Creates the fairly plain HTML folder index including a preview of a single image file,
    if any is present.

    Args:
        root (str): path to the root directory, all paths in [dirs] and
            [files] are relative to this root folder
        dirs (list): list of strings, the directories in [root]
        files (list): list of strings, the files in [root]
        dirname (str, optional): name to print in the html,
            which may be different than [root]

    Returns:
        str: HTML source of the directory listing
    """

    if dirname is None:
        dirname = root or '/'
    dirname = dirname.replace('\\','/')

    html = "<!DOCTYPE html>\n"
    html += "<html lang='en'><head>"
    html += "<title>Index of {}</title>\n".format(dirname)
    html += "<meta charset='UTF-8'>\n"
    html += "<style>\n"
    html += "body { font-family: Segoe UI, Helvetica, Arial, sans-serif; }\n"
    html += "a { text-decoration:none; }\n"
    html += "</style>\n"
    html += "</head><body>\n"

    html += "<h1>Index of {}</h1>\n".format(dirname)

    # Insert preview image
    jpg_files = [f for f in files if is_image_file(f)]

    if len(jpg_files) > 0:

        # This is slow, so defaulting to just the first image:
        #
        # Use the largest image file as this is most likely to contain
        # useful content.
        #
        # jpg_file_sizes = [os.path.getsize(f) for f in jpg_files]
        # largest_file_index = max(range(len(jpg_files)), key=lambda x: jpg_file_sizes[x])

        html += "<a href='{0}'><img style='height:200px; float:right;' src='{0}' alt='Preview image'></a>\n".\
            format(jpg_files[0])
    else:
        html += "\n"
        # html += "<p style='width:15em; float:right; margin:0;'>[No preview available]</p>\n"

    if root:
        html += "<p><a href='../index.html'>To parent directory</a></p>\n"
    else:
        html += "\n"
        # html += "<p>This is the root directory.</p>\n"

    html += "<h2>Folders</h2>\n"
    if len(dirs) > 0:
        html += "<ul style='list-style-type: none; padding-left:1em;'>\n"
        for dir in sorted(dirs):
            html += "<li>&#128193; <a href='{0}/index.html'>{0}</a></li>\n".format(dir)
        html += "</ul>\n"
    else:
        html += "<p style='padding-left:1em;'>No folders</p>\n"

    html += "<h2>Files</h2>\n"
    if len(files) > 0:
        html += "<ul style='list-style-type: none; padding-left:1em;'>\n"
        for fname in sorted(files):
            if is_image_file(fname):
                html += "<li>&#128443; <a href='{0}'>{0}</a></li>\n".format(fname)
            else:
                html += "<li>&#128442; <a href='{0}'>{0}</a></li>\n".format(fname)
        html += "</ul>\n"
    else:
        html += "<p style = 'padding-left:1em;'>No files</p>\n"

    # Add some space at the bottom because the browser's status bar might hide stuff
    html += "<p style='margin:2em;'>&nbsp;</p>\n"
    html += "</body></html>\n"
    return html

# ...def _create_plain_index(...)


def create_html_index(dir,
                      overwrite=False,
                      template_fun=_create_plain_index,
                      basepath=None,
                      recursive=True):
    """
    Recursively traverses the local directory [dir] and generates a index
    file for each folder using [template_fun] to generate the HTML output.
    Excludes hidden files.

    Args:
        dir (str): directory to process
        overwrite (bool, optional): whether to over-write existing index file
        template_fun (func, optional): function taking three arguments (string,
            list of string, list of string) representing the current root, the list of folders,
            and the list of files.  Should return the HTML source of the index file.
        basepath (str, optional): if not None, the name used for each subfolder in [dir]
            in the output files will be relative to [basepath]
        recursive (bool, optional): recurse into subfolders
    """

    if template_fun is None:
        template_fun = _create_plain_index

    print('Traversing {}'.format(dir))

    # Make sure we remove the trailing /
    dir = os.path.normpath(dir)

    # Traverse directory and all sub directories, excluding hidden files
    for root, dirs, files in os.walk(dir):

        # Exclude files and folders that are hidden
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        # Output is written to file *root*/index.html
        output_file = os.path.join(root, "index.html")

        if (not overwrite) and os.path.isfile(output_file):
            print('Skipping {}, file exists'.format(output_file))
            continue

        print("Generating {}".format(output_file))

        # Generate HTML with template function
        dirname = None
        if basepath is not None:
            dirname = os.path.relpath(root,basepath)
        html = template_fun(root[len(dir):], dirs, files, dirname)

        # Write to file
        with open(output_file, 'wt') as fi:
            fi.write(html)

        if not recursive:
            break

# ...def create_html_index(...)


#%% Command-line driver

def main(): # noqa

    parser = argparse.ArgumentParser()

    parser.add_argument("directory", type=str,
                        help='Path to directory which should be traversed.')
    parser.add_argument("--basepath", type=str,
                        help='Folder names will be printed relative to basepath, if specified',
                        default=None)
    parser.add_argument("--overwrite", action='store_true', default=False,
                        help='If set, the script will overwrite existing index.html files.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.isdir(args.directory), "{} is not a valid directory".format(args.directory)

    create_html_index(args.directory,
                      overwrite=args.overwrite,
                      basepath=args.basepath)

if __name__ == '__main__':
    main()
