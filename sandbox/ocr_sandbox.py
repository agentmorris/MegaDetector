########
#
# ocr_sandbox.py
#
# Use OCR (via the Tesseract package) to pull metadata from camera trap images.
#
# The general approach is:
#
# * Crop a fixed percentage from the top and bottom of an image, slightly larger
#   than the largest examples we've seen of how much space is used for metadata.
#
# * Refine that crop by blurring a little, then looking for huge peaks in the 
#   color histogram suggesting a solid background, then finding rows that are
#   mostly that color.
#
# * Crop to the refined crop, then run pytesseract to extract text
#
# * Use regular expressions to find time and date
#
# Prior to using this module:
#
# * Install Tesseract from https://tesseract-ocr.github.io/tessdoc/Installation.html
#
# * pip install pytesseract
#    
########

#%% Notes to self

"""

* To use the legacy engine (--oem 0), I had to download an updated eng.traineddata file from:
    
  https://github.com/tesseract-ocr/tessdata
  
"""

#%% Constants and imports

import os
import json
import numpy as np
import datetime

import cv2
from PIL import Image
from tqdm import tqdm

from md_utils.path_utils import find_images
from md_visualization import visualization_utils as vis_utils
from md_utils import write_html_image_list        

# pip install dateparser
import dateparser

# pip install datefinder
import datefinder

from dateparser.search import search_dates # noqa

# pip install pytesseract
#
# Also install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki, and add
# the installation dir to your path (on Windows, typically C:\Program Files (x86)\Tesseract-OCR)
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Using a semi-arbitrary metric of how much it feels like we found the 
# text-containing region, discard regions that appear to be extraction failures
p_crop_success_threshold = 0.5

# Pad each crop with a few pixels to make tesseract happy
crop_padding = 10        

# Discard short text, typically text from the top of the image
min_text_length = 4

# When we're looking for pixels that match the background color, allow some 
# tolerance around the dominant color
background_tolerance = 2
    
# We need to see a consistent color in at least this fraction of pixels in our rough 
# crop to believe that we actually found a candidate metadata region.
min_background_fraction = 0.3

# What fraction of the [top,bottom] of the image should we use for our rough crop?
image_crop_fraction = [0.045 , 0.045]

# A row is considered a probable metadata row if it contains at least this fraction
# of the background color.  This is used only to find the top and bottom of the crop area, 
# so it's not that *every* row needs to hit this criteria, only the rows that are generally
# above and below the text.
min_background_fraction_for_background_row = 0.5

# psm 6: "assume a single uniform block of text"
# psm 13: raw line
# oem: 0 == legacy, 1 == lstm
# tesseract_config_string = '--oem 0 --psm 6'
tesseract_config_string = '--oem 1 --psm 6'

dateparser_settings = {'PREFER_DATES_FROM':'past','STRICT_PARSING':True}


#%% Support functions

def crop_to_solid_region(image):
    """   
    cropped_image,p_success,padded_image = crop_to_solid_region(image)
    
    image should be a numpy array.
    
    Within a region of an image (typically a crop from the top-ish or bottom-ish part of 
    an image), tightly crop to the solid portion (typically a region with a black background).

    The success metric is just a binary indicator right now: 1.0 if we found a region we believe
    contains a solid background, 0.0 otherwise.
    """
           
    analysis_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)        
    analysis_image = analysis_image.astype('uint8')     
    analysis_image = cv2.medianBlur(analysis_image,3) 
    pixel_values = analysis_image.flatten()
    counts = np.bincount(pixel_values)
    background_value = int(np.argmax(counts))
    
    # Did we find a sensible mode that looks like a background value?
    background_value_count = int(np.max(counts))
    p_background_value = background_value_count / np.sum(counts)
    
    # This looks very scientific, right?  Definitely a probability?
    if (p_background_value < min_background_fraction):
        p_success = 0.0
    else:
        p_success = 1.0
        
    analysis_image = cv2.inRange(analysis_image,
                                background_value-background_tolerance,
                                background_value+background_tolerance)
    
    # Notes to self, things I tried that didn't really go anywhere...
    #
    # analysis_image = cv2.blur(analysis_image, (3,3))
    # analysis_image = cv2.medianBlur(analysis_image,5) 
    # analysis_image = cv2.Canny(analysis_image,100,100)
    # image_pil = Image.fromarray(analysis_image); image_pil
    
    # Use row heuristics to refine the crop        
    h = analysis_image.shape[0]
    w = analysis_image.shape[1]
    
    min_x = 0
    min_y = -1
    max_x = w
    max_y = -1
    
    for y in range(h):
        row_count = 0
        for x in range(w):
            if analysis_image[y][x] > 0:
                row_count += 1
        row_fraction = row_count / w
        if row_fraction > min_background_fraction_for_background_row:
            if min_y == -1:
                min_y = y
            max_y = y
    
    x = min_x
    y = min_y
    w = max_x-min_x
    h = max_y-min_y
    
    x = min_x
    y = min_y
    w = max_x-min_x
    h = max_y-min_y

    # Crop the image
    cropped_image = image[y:y+h,x:x+w]
      
    # Tesseract doesn't like characters really close to the edge, so pad a little.
    padded_crop = cv2.copyMakeBorder(cropped_image,crop_padding,crop_padding,crop_padding,crop_padding,
                                     cv2.BORDER_CONSTANT,
                                     value=[background_value,background_value,background_value])

    return cropped_image,p_success,padded_crop
    
# ...crop_to_solid_region(...)    


def rough_crop(image):
    """
    Crops the top and bottom regions out of an image, returns those as a length-two list of
    images.
    
    [image] can be a PIL image or a file name.
    """
    
    if isinstance(image,str):
        image = vis_utils.open_image(image)
        
    w = image.width
    h = image.height
    
    crop_height_top = round(image_crop_fraction[0] * h)
    crop_height_bottom = round(image_crop_fraction[1] * h)
    
    # l,t,r,b
    #
    # 0,0 is upper-left
    top_crop = image.crop([0,0,w,crop_height_top])
    bottom_crop = image.crop([0,h-crop_height_bottom,w,h])
    return [top_crop,bottom_crop]
    
# ...def rough_crop(...)


def find_text_in_crops(crops):
    """
    Find all text in each Image in the list [crops]; those images should be pretty small 
    regions by the time they get to this function, roughly the top or bottom 20% of an image.
    
    Returns a dict with fields "text_results" and "padded_crops", each the same length as [crops].
    """
    
    text_results = []
    padded_crops = []
    
    for i_crop,crop in enumerate(crops):

        # pil --> cv2        
        crop = np.array(crop) 
        crop = crop[:, :, ::-1].copy()         
        
        # image = cv2.medianBlur(image, 3)        
        # image = cv2.erode(image, None, iterations=2)
        # image = cv2.dilate(image, None, iterations=4)
        # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # image = cv2.blur(image, (3,3))
        # image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        crop,p_success,padded_crop = crop_to_solid_region(crop)
        
        if p_success < p_crop_success_threshold:
            text_results.append('')
            continue
        
        padded_crop_pil = Image.fromarray(padded_crop)
        padded_crops.append(padded_crop_pil)
        
        # text = pytesseract.image_to_string(image_pil, lang='eng')
        # https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
        
        text = pytesseract.image_to_string(padded_crop_pil, lang='eng', config=tesseract_config_string)
        text = text.replace('\n', ' ').replace('\r', '').strip()

        text_results.append(text)
                
    # ...for each cropped region
    
    return {'text_results':text_results,
            'padded_crops':padded_crops}
    
# ...def find_text_in_crops(...)
    

def get_datetime_from_strings(strings):
    """
    Given a list of strings, search for exactly one datetime in those strings. 
    
    Strings are currently just concatenated before searching for a datetime.
    """
    
    s = ' '.join(strings)    
        
    dateparser_result = dateparser.search.search_dates(s, settings=dateparser_settings)
    
    if dateparser_result is not None:
        assert len(dateparser_result) == 1
        extracted_datetime = dateparser_result[0][1]
    else:
        matches = datefinder.find_dates(s,strict=False)
        matches_list = [m for m in matches]
        if len(matches_list) == 1:
            extracted_datetime = matches_list[0]
        else:
            extracted_datetime = None
        
    if extracted_datetime is not None:        
        assert extracted_datetime.year <= 2023 and extracted_datetime.year >= 1990
    
    return extracted_datetime
    
# ...def get_datetime_from_strings(...)


def get_datetime_from_image(image,include_crops=True):
    """
    Find the datetime string (if present) in [image], which can be a PIL image or a 
    filename.  Returns a dict:
        
    text_results: length-2 list of strings
    crops: length-2 list of images
    padded_crops: length-2 list of images
    datetime: Python datetime object, or None
    """
    
    if isinstance(image,str):
        image = vis_utils.open_image(image)

    # Crop the top and bottom from the image
    crops = rough_crop(image)
    assert len(crops) == 2
    
    # Find text
    ocr_results = find_text_in_crops(crops)
    text_results = ocr_results['text_results']
    padded_crops = ocr_results['padded_crops']
    assert len(text_results) == 2
    assert len(padded_crops) == 2
        
    # Find datetime
    extracted_datetime = get_datetime_from_strings(text_results)
    assert isinstance(extracted_datetime,datetime.datetime) or (extracted_datetime is None)
    
    to_return = {}
    to_return['text_results'] = text_results
    to_return['datetime'] = extracted_datetime
    
    if include_crops:
        to_return['crops'] = crops
        to_return['padded_crops'] = padded_crops
        
    return to_return

# ...def get_datetime_from_image(...)


def try_get_datetime_from_image(filename,include_crops=False):
    """
    Try/catch wrapper for get_datetime_from_image, defaults to returning 
    metadata only.
    """
        
    result = {}
    result['error'] = None        
    try:
        result = get_datetime_from_image(filename,include_crops)
    except Exception as e:
        result['error'] = str(e)
    return result


def get_datetimes_for_folder(folder_name,output_file,n_to_sample=-1):
    """
    Retrieve metadata from every image in [folder_name], and 
    write the results to the .json file [output_file].
    """
    
    image_file_names = \
        find_images(folder_name,convert_slashes=True,
                    return_relative_paths=False,recursive=True)
    
    if n_to_sample > 0:
        import random
        random.seed(0)
        image_file_names = random.sample(image_file_names,n_to_sample)
        
    n_cores = 8
    use_threads = False
    
    if n_cores <= 1:
        all_results = []
        for fn_abs in tqdm(image_file_names):
            all_results.append(try_get_datetime_from_image(fn_abs))
    else:    
        
        if use_threads:
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(n_cores)
            worker_string = 'threads'        
        else:
            from multiprocessing.pool import Pool
            pool = Pool(n_cores)
            worker_string = 'processes'
            
        print('Starting a pool of {} {}'.format(n_cores,worker_string))
        
        all_results = list(tqdm(pool.imap(
            try_get_datetime_from_image,image_file_names), total=len(image_file_names)))
    
    filename_to_results = {}
    
    # fn_relative = image_file_names[0]
    for i_file,fn_abs in enumerate(image_file_names):
        filename_to_results[fn_abs] = all_results[i_file]
    
    with open(output_file,'w') as f:
        json.dump(filename_to_results,f,indent=1,default=str)


#%% Interactive driver

if False:
    
    #%% Process images
    
    folder_name = r'g:\temp\island_conservation_camera_traps'
    output_file = r'g:\temp\ocr_results.json'
    assert os.path.isdir(folder_name)
    get_datetimes_for_folder(folder_name,output_file)
    

    #%% Write results to an HTML file for testing
          
    with open(output_file,'r') as f:
        filename_to_results = json.load(f)
        
    preview_dir = r'g:\temp\ocr-preview'
    os.makedirs(preview_dir,exist_ok=True)
    output_summary_file = os.path.join(preview_dir,'summary.html')
    
    html_image_list = []
    html_title_list = []
    
    html_options = write_html_image_list.write_html_image_list()
    
    # i_image = 0; fn_relative = next(iter(filename_to_results))
    for i_image,fn_abs in tqdm(enumerate(filename_to_results),total=len(filename_to_results)):
            
        fn_relative = os.path.relpath(fn_abs,folder_name)
        
        # Add image name and resized image
        resized_image = vis_utils.resize_image(fn_abs,target_width=600)
        resized_fn = os.path.join(preview_dir,'img_{}_base.png'.format(i_image))
        resized_image.save(resized_fn)
        
        results_this_image = filename_to_results[fn_abs]
            
        extracted_datetime = results_this_image['datetime']
        title = 'Image: {}<br/>Extracted datetime: {}'.format(fn_relative,extracted_datetime)
        html_image_list.append({'filename':resized_fn,'title':title})
                
        for i_crop,crop in enumerate(results_this_image['crops']):
                
            image = vis_utils.resize_image(crop,target_width=600)
            fn_crop = os.path.join(preview_dir,'img_{}_r{}_processed.png'.format(i_image,i_crop))
            image.save(fn_crop)
            
            image_style = html_options['defaultImageStyle'] + 'margin-left:50px;'
            text_style = html_options['defaultTextStyle'] + 'margin-left:50px;'
            
            if i_crop == len(results_this_image['crops']) - 1:
                image_style += 'margin-bottom:30px;'
                
            title = 'Raw text: ' + results_this_image['text_results'][i_crop]
            
            # textStyle = "font-family:calibri,verdana,arial;font-weight:bold;font-size:150%;text-align:left;margin-left:50px;"
            html_image_list.append({'filename':fn_crop,'imageStyle':image_style,'title':title,'textStyle':text_style})
                            
        # ...for each crop
    
    # ...for each image
            
    html_options['makeRelative'] = True
    write_html_image_list.write_html_image_list(output_summary_file,
                                                html_image_list,
                                                html_options)
    from md_utils.path_utils import open_file
    open_file(output_summary_file)


#%% Alternative approaches to finding the text/background region

if False:
    
    #%% Using findContours()

    # image_pil = Image.fromarray(analysis_image); image_pil
    
    # analysis_image = cv2.erode(analysis_image, None, iterations=3)
    # analysis_image = cv2.dilate(analysis_image, None, iterations=3)

    # analysis_image = cv2.threshold(analysis_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    im2, contours, hierarchy = cv2.findContours(analysis_image,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) # noqa
    
    # Find object with the biggest bounding box
    mx = (0,0,0,0)      # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x,y,w,h = cv2.boundingRect(cont)
        area = w*h
        if area > mx_area:
            mx = x,y,w,h
            mx_area = area
    x,y,w,h = mx
   
    
   #%% Using connectedComponents()
    
    # analysis_image = image
    nb_components, output, stats, centroids = \
        cv2.connectedComponentsWithStats(analysis_image, connectivity = 4) # noqa
    # print('Found {} components'.format(nb_components))
    sizes = stats[:, -1]

    max_label = 1
    max_size = sizes[1]
    for i in range(2, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    # We just want the *background* image
    max_label = 0
    
    mask_image = np.zeros(output.shape)
    mask_image[output == max_label] = 255
    
    thresh = 127
    binary_image = cv2.threshold(mask_image, thresh, 255, cv2.THRESH_BINARY)[1]
    
    min_x = -1
    min_y = -1
    max_x = -1
    max_y = -1
    h = binary_image.shape[0]
    w = binary_image.shape[1]
    for y in range(h):
        for x in range(w):
            if binary_image[y][x] > thresh:
                if min_x == -1:
                    min_x = x
                if min_y == -1:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y
