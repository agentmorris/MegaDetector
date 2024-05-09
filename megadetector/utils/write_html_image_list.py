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
            
            - fHtml (file pointer to write to, used for splitting write operations over multiple calls)
            - headerHtml (html text to include before the image list)
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
        
    if 'fHtml' not in options:
        options['fHtml'] = -1
    
    if 'headerHtml' not in options or options['headerHtml'] is None:
        options['headerHtml'] = ''        
    
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
    for iImage,imageInfo in enumerate(images):
        if isinstance(imageInfo,str):
            imageInfo = {'filename':imageInfo}
        if 'filename' not in imageInfo:
            imageInfo['filename'] = ''
        if 'imageStyle' not in imageInfo:
            imageInfo['imageStyle'] = options['defaultImageStyle']
        if 'title' not in imageInfo:
            imageInfo['title'] = ''
        if 'linkTarget' not in imageInfo:
            imageInfo['linkTarget'] = ''
        if 'textStyle' not in imageInfo:
            textStyle = options['defaultTextStyle']
            imageInfo['textStyle'] = options['defaultTextStyle']
        images[iImage] = imageInfo            
    
    nImages = len(images)
    
    # If we need to break this up into multiple files...
    if nImages > options['maxFiguresPerHtmlFile']:
    
        # You can't supply your own file handle in this case
        if options['fHtml'] != -1:
            raise ValueError(
                    'You can''t supply your own file handle if we have to page the image set')
        
        figureFileStartingIndices = list(range(0,nImages,options['maxFiguresPerHtmlFile']))

        assert len(figureFileStartingIndices) > 1
        
        # Open the meta-output file
        fMeta = open(filename,'w')
        
        # Write header stuff
        fMeta.write('<html><body>\n')    
        fMeta.write(options['headerHtml'])        
        fMeta.write('<table border = 0 cellpadding = 2>\n')
        
        for startingIndex in figureFileStartingIndices:
            
            iStart = startingIndex
            iEnd = startingIndex+options['maxFiguresPerHtmlFile']-1;
            if iEnd >= nImages:
                iEnd = nImages-1
            
            trailer = 'image_{:05d}_{:05d}'.format(iStart,iEnd)
            localFiguresHtmlFilename = path_utils.insert_before_extension(filename,trailer)
            fMeta.write('<tr><td>\n')
            fMeta.write('<p style="padding-bottom:0px;margin-bottom:0px;text-align:left;font-family:''segoe ui'',calibri,arial;font-size:100%;text-decoration:none;font-weight:bold;">')
            fMeta.write('<a href="{}">Figures for images {} through {}</a></p></td></tr>\n'.format(
                os.path.basename(localFiguresHtmlFilename),iStart,iEnd))
            
            localImages = images[iStart:iEnd+1]
            
            localOptions = options.copy();
            localOptions['headerHtml'] = '';
            localOptions['trailerHtml'] = '';
            
            # Make a recursive call for this image set
            write_html_image_list(localFiguresHtmlFilename,localImages,localOptions)
            
        # ...for each page of images
        
        fMeta.write('</table></body>\n')
        fMeta.write(options['trailerHtml'])
        fMeta.write('</html>\n')
        fMeta.close()
        
        return options
        
    # ...if we have to make multiple sub-pages
        
    bCleanupFile = False
    
    if options['fHtml'] == -1:
        bCleanupFile = True;
        fHtml = open(filename,'w')
    else:
        fHtml = options['fHtml']
        
    fHtml.write('<html><body>\n')
    
    fHtml.write(options['headerHtml'])
    
    # Write out images
    for iImage,image in enumerate(images):
        
        title = image['title']
        imageStyle = image['imageStyle']
        textStyle = image['textStyle']
        filename = image['filename']
        linkTarget = image['linkTarget']
        
        # Remove unicode characters
        title = title.encode('ascii','ignore').decode('ascii')
        filename = filename.encode('ascii','ignore').decode('ascii')
        
        filename = filename.replace('\\','/')
        if options['urlEncodeFilenames']:            
            filename = urllib.parse.quote(filename)
        
        if len(title) > 0:       
            fHtml.write(
                    '<p style="{}">{}</p>\n'\
                    .format(textStyle,title))            

        linkTarget = linkTarget.replace('\\','/')
        if options['urlEncodeLinkTargets']:
            # These are typically absolute paths, so we only want to mess with certain characters
            linkTarget = urllib.parse.quote(linkTarget,safe=':/')
            
        if len(linkTarget) > 0:
            fHtml.write('<a href="{}">'.format(linkTarget))
            # imageStyle.append(';border:0px;')
        
        fHtml.write('<img src="{}" style="{}">\n'.format(filename,imageStyle))
        
        if len(linkTarget) > 0:
            fHtml.write('</a>')
            
        if iImage != len(images)-1:
            fHtml.write('<br/>')             
            
    # ...for each image we need to write
    
    fHtml.write(options['trailerHtml'])
    
    fHtml.write('</body></html>\n')
    
    if bCleanupFile:
        fHtml.close()    

# ...function
