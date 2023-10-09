########
#
# directory_listing.py
#
# Script for creating HTML directory listings for a local directory and
# all its subdirectories.  Primarily intended for use on mounted blob
# containers, so it includes the ability to set the content-type property
# on the generated html using the blob SDK, so it can be browser-viewable.
#
# Also includes a preview of a jpg file (the first in an alphabetical list),
# if present.
#
# Sample invocation:
#
# python directory_listing.py /naipout/v002 --basepath /naipout/v002 \
#   --enable_overwrite \
#   --sas_url "https://naipblobs.blob.core.windows.net/naip/v002?sv=..."
#
########

#%% Imports

import os
import sys
import argparse
import re

import azure.common
from azure.storage.blob import BlobServiceClient, ContentSettings


#%% Directory enumeration functions

def is_image_file(path):
    """
    Checks whether the provided file path points to an image by checking the
    file extension. The following file extensions are considered images: jpg,
    jpeg.  The check is case-insensitive.

    Args:
        path: string, path to image file or just the file name

    Returns:
        boolean, True if the file is an image
    """

    return os.path.splitext(path)[1].lower()[1:] in ['jpg', 'jpeg'] #, 'gif', 'tiff', 'tif', 'png']


def create_plain_index(root, dirs, files, dirname=None):
    """
    Creates the fairly plain HTML folder index
    including a preview of a single image file, if any is present.
    Returns the HTML source as string.

    Args:
        root: string, path to the root directory, all paths in *dirs* and
            *files* are relative to this one
        dirs: list of strings, the directories in *root*
        files: list of string, the files in *root*
        dirname: name to print in the html, which may be different than *root*

    Returns: HTML source of the directory listing
    """

    if dirname is None:
        dirname = root or '/'

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

        html += "<a href='{0}'><img style='height:200px; float:right;' src='{0}' alt='Preview image'></a>\n".format(jpg_files[0])
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


def traverse_and_create_index(dir, sas_url=None, overwrite_files=False,
                              template_fun=create_plain_index, basepath=None):
    """
    Recursively traverses the local directory *dir* and generates a index
    file for each folder using *template_fun* to generate the HTML output.
    Excludes hidden files.

    Args:
        dir: string, path to directory
        template_fun: function taking three arguments (string, list of string, list of string)
            representing the current root, the list of folders, and the list of files.
            Should return the HTML source of the index file

    Return:
        None
    """

    print("Traversing {}".format(dir))

    # Make sure we remove the trailing /
    dir = os.path.normpath(dir)

    # If we want to set the content type in blob storage using a SAS URL
    if sas_url:

        # Example: sas_url = 'https://accname.blob.core.windows.net/bname/path/to/folder?st=...&se=...&sp=...&...'
        if '?' in sas_url:
            # 'https://accname.blob.core.windows.net/bname/path/to/folder' and 'st=...&se=...&sp=...&...'
            base_url, sas_token = sas_url.split('?', 1)
        else:
            # 'https://accname.blob.core.windows.net/bname/path/to/folder' and None
            base_url, sas_token = sas_url, None
        # Remove https:// from base url
        # 'accname.blob.core.windows.net/bname/path/to/folder'
        base_url = base_url.split("//", 1)[1]
        # Everything up to the first dot is account name
        # 'accname'
        account_name = base_url.split(".", 1)[0]
        # get everything after the first /
        # 'bname/path/to/folder'
        query_string = base_url.split("/", 1)[1]
        # Get container name and subfolder
        if '/' in query_string:
            # 'bname', 'path/to/folder'
            container_name, container_folder = query_string.split("/", 1)
        else:
            container_name, container_folder = query_string, ''

        # Prepare the storage access
        target_settings = ContentSettings(content_type='text/html')
        blob_service = BlobServiceClient(
            account_url=f'{account_name}.blob.core.windows.net',
            credential=sas_token)

    # Traverse directory and all sub directories, excluding hidden files
    for root, dirs, files in os.walk(dir):

        # Exclude files and folders that are hidden
        files = [f for f in files if not f[0] == '.']
        dirs[:] = [d for d in dirs if not d[0] == '.']

        # Output is written to file *root*/index.html
        output_file = os.path.join(root, "index.html")

        if not overwrite_files and os.path.isfile(output_file):
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

        # Set content type in blob storage
        if sas_url:
            if container_folder:
                output_blob_path = container_folder + '/' + output_file[len(dir) + 1:]
            else:
                output_blob_path = output_file[len(dir) + 1:]
            try:
                blob_client = blob_service.get_blob_client(container_name, output_blob_path)
                blob_client.set_http_headers(content_settings=target_settings)
            except azure.common.AzureMissingResourceHttpError:
                print('ERROR: It seems the SAS URL is incorrect or does not allow setting properties.')
                return


#%% Command-line driver

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help='Path to directory which should be traversed.')
    parser.add_argument("--basepath", type=str, help='Folder names will be printed relative to basepath, if specified', default=None)
    parser.add_argument("--sas_url", type=str, help='Blobfuse does not set the content-type property ' + \
        'properly and hence index.html won\'t be accessible in the browser. If you want to set the ' + \
        'content-type in the corresponding blob storage, provide the SAS URL that corresponds to the ' + \
        "directory, e.g. if *directory* is /mountpoint/path/to/folder, then *--sas_url* looks like " + \
        "'https://accname.blob.core.windows.net/bname/path/to/folder?st=...&se=...&sp=...&...'")
    parser.add_argument("--enable_overwrite", action='store_true', default=False,
                        help='If set, the script will overwrite existing index.html files.')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.isdir(args.directory), "{} is not a valid directory".format(args.directory)
    assert re.match('https?://[^\.]+\.blob\.core\.windows\.net/.+', args.sas_url), "--sas_url does not " + \
        "match the format https://accname.blob.core.windows.net/bname/path/to/folder?..."

    traverse_and_create_index(args.directory, overwrite_files=args.enable_overwrite, sas_url=args.sas_url, basepath=args.basepath)
