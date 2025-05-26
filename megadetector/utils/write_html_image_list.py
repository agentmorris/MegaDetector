"""

write_html_image_list.py

Given a list of image file names, writes an HTML file that
shows all those images, with optional one-line headers above each.

Each "filename" can also be a dict with elements 'filename','title',
'imageStyle','textStyle', 'linkTarget'

"""

#%% Constants and imports

import os
import math
import urllib

from megadetector.utils import path_utils


#%% write_html_image_list

def write_html_image_list(filename=None,images=None,options=None):
    """
    Given a list of image file names, writes an HTML file that shows all those images,
    with optional one-line headers above each.

    Args:
        filename (str, optional): the .html output file; if None, just returns a valid
            options dict
        images (list, optional): the images to write to the .html file; if None, just returns
            a valid options dict.  This can be a flat list of image filenames, or this can
            be a list of dictionaries with one or more of the following fields:

            - filename (image filename) (required, all other fields are optional)
            - imageStyle (css style for this image)
            - textStyle (css style for the title associated with this image)
            - title (text label for this image)
            - linkTarget (URL to which this image should link on click)

        options (dict, optional): a dict with one or more of the following fields:

            - f_html (file pointer to write to, used for splitting write operations over multiple calls)
            - pageTitle (HTML page title)
            - headerHtml (html text to include before the image list)
            - subPageHeaderHtml (html text to include before the images when images are broken into pages)
            - trailerHtml (html text to include after the image list)
            - defaultImageStyle (default css style for images)
            - defaultTextStyle (default css style for image titles)
            - maxFiguresPerHtmlFile (max figures for a single HTML file; overflow will be handled by creating
              multiple files and a TOC with links)
            - urlEncodeFilenames (default True, e.g. '#' will be replaced by '%23')
            - urlEncodeLinkTargets (default True, e.g. '#' will be replaced by '%23')

    """

    # returns an options struct
    if options is None:
        options = {}

    if 'f_html' not in options:
        options['f_html'] = -1

    if 'pageTitle' not in options or options['pageTitle'] is None:
        options['pageTitle'] = ''

    if 'headerHtml' not in options or options['headerHtml'] is None:
        options['headerHtml'] = ''

    if 'subPageHeaderHtml' not in options or options['subPageHeaderHtml'] is None:
        options['subPageHeaderHtml'] = ''

    if 'trailerHtml' not in options or options['trailerHtml'] is None:
        options['trailerHtml'] = ''

    if 'defaultTextStyle' not in options or options['defaultTextStyle'] is None:
        options['defaultTextStyle'] = \
        "font-family:calibri,verdana,arial;font-weight:bold;font-size:150%;text-align:left;margin:0px;"

    if 'defaultImageStyle' not in options or options['defaultImageStyle'] is None:
        options['defaultImageStyle'] = \
        "margin:0px;margin-top:5px;margin-bottom:5px;"

    if 'urlEncodeFilenames' not in options or options['urlEncodeFilenames'] is None:
        options['urlEncodeFilenames'] = True

    if 'urlEncodeLinkTargets' not in options or options['urlEncodeLinkTargets'] is None:
        options['urlEncodeLinkTargets'] = True

    # Possibly split the html output for figures into multiple files; Chrome gets sad with
    # thousands of images in a single tab.
    if 'maxFiguresPerHtmlFile' not in options or options['maxFiguresPerHtmlFile'] is None:
        options['maxFiguresPerHtmlFile'] = math.inf

    if filename is None or images is None:
        return options

    # images may be a list of images or a list of image/style/title dictionaries,
    # enforce that it's the latter to simplify downstream code
    for i_image,image_info in enumerate(images):
        if isinstance(image_info,str):
            image_info = {'filename':image_info}
        if 'filename' not in image_info:
            image_info['filename'] = ''
        if 'imageStyle' not in image_info:
            image_info['imageStyle'] = options['defaultImageStyle']
        if 'title' not in image_info:
            image_info['title'] = ''
        if 'linkTarget' not in image_info:
            image_info['linkTarget'] = ''
        if 'textStyle' not in image_info:
            text_style = options['defaultTextStyle']
            image_info['textStyle'] = options['defaultTextStyle']
        images[i_image] = image_info

    n_images = len(images)

    # If we need to break this up into multiple files...
    if n_images > options['maxFiguresPerHtmlFile']:

        # You can't supply your own file handle in this case
        if options['f_html'] != -1:
            raise ValueError(
                    "You can't supply your own file handle if we have to page the image set")

        figure_file_starting_indices = list(range(0,n_images,options['maxFiguresPerHtmlFile']))

        assert len(figure_file_starting_indices) > 1

        # Open the meta-output file
        f_meta = open(filename,'w')

        # Write header stuff
        title_string = '<title>Index page</title>'
        if len(options['pageTitle']) > 0:
            title_string = '<title>Index page for: {}</title>'.format(options['pageTitle'])
        f_meta.write('<html><head>{}</head><body>\n'.format(title_string))
        f_meta.write(options['headerHtml'])
        f_meta.write('<table border = 0 cellpadding = 2>\n')

        for starting_index in figure_file_starting_indices:

            i_start = starting_index
            i_end = starting_index + options['maxFiguresPerHtmlFile'] - 1
            if i_end >= n_images:
                i_end = n_images-1

            trailer = 'image_{:05d}_{:05d}'.format(i_start,i_end)
            local_figures_html_filename = path_utils.insert_before_extension(filename,trailer)
            f_meta.write('<tr><td>\n')
            f_meta.write('<p style="padding-bottom:0px;margin-bottom:0px;text-align:left;font-family:''segoe ui'',calibri,arial;font-size:100%;text-decoration:none;font-weight:bold;">') # noqa
            f_meta.write('<a href="{}">Figures for images {} through {}</a></p></td></tr>\n'.format(
                os.path.basename(local_figures_html_filename),i_start,i_end))

            local_images = images[i_start:i_end+1]

            local_options = options.copy()
            local_options['headerHtml'] = options['subPageHeaderHtml']
            local_options['trailerHtml'] = ''

            # Make a recursive call for this image set
            write_html_image_list(local_figures_html_filename,local_images,local_options)

        # ...for each page of images

        f_meta.write('</table></body>\n')
        f_meta.write(options['trailerHtml'])
        f_meta.write('</html>\n')
        f_meta.close()

        return options

    # ...if we have to make multiple sub-pages

    b_clean_up_file = False

    if options['f_html'] == -1:
        b_clean_up_file = True
        f_html = open(filename,'w')
    else:
        f_html = options['f_html']

    title_string = ''
    if len(options['pageTitle']) > 0:
        title_string = '<title>{}</title>'.format(options['pageTitle'])

    f_html.write('<html>{}<body>\n'.format(title_string))

    f_html.write(options['headerHtml'])

    # Write out images
    for i_image,image in enumerate(images):

        title = image['title']
        image_style = image['imageStyle']
        text_style = image['textStyle']
        filename = image['filename']
        link_target = image['linkTarget']

        # Remove unicode characters
        title = title.encode('ascii','ignore').decode('ascii')
        filename = filename.encode('ascii','ignore').decode('ascii')

        filename = filename.replace('\\','/')
        if options['urlEncodeFilenames']:
            filename = urllib.parse.quote(filename)

        if len(title) > 0:
            f_html.write(
                    '<p style="{}">{}</p>\n'\
                    .format(text_style,title))

        link_target = link_target.replace('\\','/')
        if options['urlEncodeLinkTargets']:
            # These are typically absolute paths, so we only want to mess with certain characters
            link_target = urllib.parse.quote(link_target,safe=':/')

        if len(link_target) > 0:
            f_html.write('<a href="{}">'.format(link_target))
            # image_style.append(';border:0px;')

        f_html.write('<img src="{}" style="{}">\n'.format(filename,image_style))

        if len(link_target) > 0:
            f_html.write('</a>')

        if i_image != len(images)-1:
            f_html.write('<br/>')

    # ...for each image we need to write

    f_html.write(options['trailerHtml'])

    f_html.write('</body></html>\n')

    if b_clean_up_file:
        f_html.close()

# ...function
