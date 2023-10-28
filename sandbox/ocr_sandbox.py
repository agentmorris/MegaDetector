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
import re

from dateutil.parser import parse as dateparse

import cv2
from PIL import Image
from tqdm import tqdm

from md_utils.path_utils import find_images
from md_visualization import visualization_utils as vis_utils
from md_utils import write_html_image_list        
from md_utils.path_utils import open_file

# pip install pytesseract
#
# Also install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki, and add
# the installation dir to your path (on Windows, typically C:\Program Files (x86)\Tesseract-OCR)
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


#%% Extraction options

class DatetimeExtractionOptions:

            
    def __init__(self):
        
        # Using a semi-arbitrary metric of how much it feels like we found the 
        # text-containing region, discard regions that appear to be extraction failures
        self.p_crop_success_threshold = 0.5
        
        # Pad each crop with a few pixels to make tesseract happy
        self.crop_padding = 10        
        
        # Discard short text, typically text from the top of the image
        self.min_text_length = 4
        
        # When we're looking for pixels that match the background color, allow some 
        # tolerance around the dominant color
        self.background_tolerance = 2
            
        # We need to see a consistent color in at least this fraction of pixels in our rough 
        # crop to believe that we actually found a candidate metadata region.
        self.min_background_fraction = 0.3
        
        # What fraction of the [top,bottom] of the image should we use for our rough crop?
        self.image_crop_fraction = [0.045 , 0.045]
        
        # Within that rough crop, how much should we use for determining the background color?
        self.background_crop_fraction_of_rough_crop = 0.5
        
        # A row is considered a probable metadata row if it contains at least this fraction
        # of the background color.  This is used only to find the top and bottom of the crop area, 
        # so it's not that *every* row needs to hit this criteria, only the rows that are generally
        # above and below the text.
        self.min_background_fraction_for_background_row = 0.5
        
        # psm 6: "assume a single uniform block of text"
        # psm 13: raw line
        # oem: 0 == legacy, 1 == lstm
        # tesseract_config_string = '--oem 0 --psm 6'
        #
        # Try these configuration strings in order until we find a valid datetime
        self.tesseract_config_strings = ['--oem 1 --psm 13','--oem 0 --psm 13',
                                         '--oem 1 --psm 6','--oem 0 --psm 6']


#%% Support functions

def make_rough_crops(image,options=None):
    """
    Crops the top and bottom regions out of an image, returns a dict with fields
    'top' and 'bottom', each pointing to a PIL image.
    
    [image] can be a PIL image or a file name.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
        
    if isinstance(image,str):
        image = vis_utils.open_image(image)
        
    w = image.width
    h = image.height
    
    crop_height_top = round(options.image_crop_fraction[0] * h)
    crop_height_bottom = round(options.image_crop_fraction[1] * h)
    
    # l,t,r,b
    #
    # 0,0 is upper-left
    top_crop = image.crop([0,0,w,crop_height_top])
    bottom_crop = image.crop([0,h-crop_height_bottom,w,h])
    return {'top':top_crop,'bottom':bottom_crop}
    
# ...def make_rough_crops(...)


def crop_to_solid_region(rough_crop,crop_location,options=None):
    """   
    cropped_image,p_success,padded_image = crop_to_solid_region(image)
    
    rough_crop should be PIL Image, crop_location should be 'top' or 'bottom'.
    
    Within a region of an image (typically a crop from the top-ish or bottom-ish part of 
    an image), tightly crop to the solid portion (typically a region with a black background).

    The success metric is just a binary indicator right now: 1.0 if we found a region we believe
    contains a solid background, 0.0 otherwise.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()    

    crop_to_solid_region_result = {}
    crop_to_solid_region_result['crop_pil'] = None
    crop_to_solid_region_result['padded_crop_pil'] = None
    crop_to_solid_region_result['p_success'] = None
    
    # pil --> cv2        
    rough_crop_np = np.array(rough_crop) 
    rough_crop_np = rough_crop_np[:, :, ::-1].copy()         
    
    # Search *part* of the crop for the background value (the part closest to the top or bottom
    # of the image)
    rows_to_use_for_background_search = int(rough_crop_np.shape[0] * \
                                            options.background_crop_fraction_of_rough_crop)
    
    if crop_location == 'top':
        background_search_image = rough_crop_np[0:rows_to_use_for_background_search,:,:]
    elif crop_location == 'bottom':
        background_search_image = rough_crop_np[-rows_to_use_for_background_search:,:,:]
    else:
        raise ValueError('Unrecognized crop location: {}'.format(crop_location))
                
    background_search_image = cv2.cvtColor(background_search_image, cv2.COLOR_BGR2GRAY)
    background_search_image = background_search_image.astype('uint8')     
    background_search_image = cv2.medianBlur(background_search_image,3) 
    pixel_values = background_search_image.flatten()
    counts = np.bincount(pixel_values)
    background_value = int(np.argmax(counts))
    
    # Did we find a sensible mode that looks like a background value?
    background_value_count = int(np.max(counts))
    p_background_value = background_value_count / np.sum(counts)
    
    # This looks very scientific, right?  Definitely a probability?
    if (p_background_value < options.min_background_fraction):
        p_success = 0.0
        crop_to_solid_region_result['p_success'] = p_success
        return crop_to_solid_region_result
    else:
        p_success = 1.0
        
    analysis_image = cv2.cvtColor(rough_crop_np, cv2.COLOR_BGR2GRAY)
    analysis_image = analysis_image.astype('uint8')     
    analysis_image = cv2.medianBlur(analysis_image,3) 
    
    # This will now be a binary image indicating which pixels are background
    analysis_image = cv2.inRange(analysis_image,
                                background_value-options.background_tolerance,
                                background_value+options.background_tolerance)
    
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
        if row_fraction > options.min_background_fraction_for_background_row:
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
    crop_np = rough_crop_np[y:y+h,x:x+w]
      
    # Tesseract doesn't like characters really close to the edge, so pad a little.
    crop_padding = options.crop_padding
    padded_crop_np = cv2.copyMakeBorder(crop_np,crop_padding,crop_padding,crop_padding,crop_padding,
                                     cv2.BORDER_CONSTANT,
                                     value=[background_value,background_value,background_value])

    crop_pil = Image.fromarray(crop_np)
    padded_crop_pil = Image.fromarray(padded_crop_np)
    
    crop_to_solid_region_result['crop_pil'] = crop_pil
    crop_to_solid_region_result['padded_crop_pil'] = padded_crop_pil
    crop_to_solid_region_result['p_success'] = p_success
    
    return crop_to_solid_region_result
    
# ...crop_to_solid_region(...)    


def find_text_in_crops(rough_crops,options=None,tesseract_config_string=None):
    """
    Find all text in each Image in the dict [rough_crops]; those images should be pretty small 
    regions by the time they get to this function, roughly the top or bottom 20% of an image.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    if tesseract_config_string is None:
        tesseract_config_string = options.tesseract_config_strings[0]
        
    find_text_in_crops_results = {}
    
    # crop_location = 'top'
    for crop_location in ('top','bottom'):

        find_text_in_crops_results[crop_location] = {}
        find_text_in_crops_results[crop_location]['text'] = ''
        find_text_in_crops_results[crop_location]['crop_to_solid_region_results'] = None
        
        rough_crop = rough_crops[crop_location]
                
        # image = cv2.medianBlur(image, 3)        
        # image = cv2.erode(image, None, iterations=2)
        # image = cv2.dilate(image, None, iterations=4)
        # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # image = cv2.blur(image, (3,3))
        # image = cv2.copyMakeBorder(image,10,10,10,10,cv2.BORDER_CONSTANT,value=[0,0,0])
        
        crop_to_solid_region_results = crop_to_solid_region(rough_crop,crop_location,options)
        
        if crop_to_solid_region_results['p_success'] < options.p_crop_success_threshold:
            continue
        
        find_text_in_crops_results[crop_location]['crop_to_solid_region_results'] = \
            crop_to_solid_region_results
        
        # text = pytesseract.image_to_string(image_pil, lang='eng')
        # https://github.com/tesseract-ocr/tesseract/wiki/Command-Line-Usage
        
        padded_crop_pil = crop_to_solid_region_results['padded_crop_pil']
        text = pytesseract.image_to_string(padded_crop_pil, lang='eng', 
                                           config=tesseract_config_string)
        
        text = text.replace('\n', ' ').replace('\r', '').strip()

        find_text_in_crops_results[crop_location]['text'] = text        
                
    # ...for each cropped region
    
    return find_text_in_crops_results
    
# ...def find_text_in_crops(...)
    

def datetime_string_to_datetime(matched_string):
    """
    Takes an OCR-matched datetime string, does a little cleanup, and parses a date
    from it.
    """
    
    matched_string = matched_string.replace(' -','-')
    matched_string = matched_string.replace('- ','-')
    matched_string = matched_string.replace(' :',':')
    matched_string = matched_string.replace(': ',':')
    try:
        extracted_datetime = dateparse(matched_string)
    except Exception:
        extracted_datetime = None
    return extracted_datetime


def get_datetime_from_strings(strings,options=None):
    """
    Given a string or list of strings, search for exactly one datetime in those strings. 
    
    Strings are currently just concatenated before searching for a datetime.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()    
    
    if isinstance(strings,str):
        s = strings
    else:
        s = ' '.join(strings).lower()
    s = s.replace('—','-')    
    s = ''.join(e for e in s if e.isalnum() or e in ':-' or e.isspace())
        
    # 2013-10-02 11:40:50 AM
    m = re.search('(\d\d\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d)\s+(\d+)\s?:?\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))        
    
    # 04/01/2017 08:54:00AM
    m = re.search('(\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d\d\d)\s+(\d+)\s?:\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))    
    
    # 2017/04/01 08:54:00AM
    m = re.search('(\d\d\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d)\s+(\d+)\s?:\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))        
    
    # 04/01/2017 08:54AM
    m = re.search('(\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d\d\d)\s+(\d+)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))    
    
    # 2017/04/01 08:54AM
    m = re.search('(\d\d\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d)\s+(\d+)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))        
    
    # 2013-07-27 04:56:35
    m = re.search('(\d\d\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))        
    
    # 07-27-2013 04:56:35
    m = re.search('(\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return datetime_string_to_datetime(m.group(0))        
    
    return None
    
# ...def get_datetime_from_strings(...)


def get_datetime_from_image(image,include_crops=True,options=None):
    """
    Find the datetime string (if present) in [image], which can be a PIL image or a 
    filename.  Returns a dict:
        
    text_results: length-2 list of strings
    crops: length-2 list of images
    padded_crops: length-2 list of images
    datetime: Python datetime object, or None
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    if isinstance(image,str):
        image = vis_utils.open_image(image)

    # Crop the top and bottom from the image
    rough_crops = make_rough_crops(image,options)
    assert len(rough_crops) == 2
    
    # Find text, trying config strings until we match a datetime
    #
    # tesseract_config_string = options.tesseract_config_strings[0]
    for tesseract_config_string in options.tesseract_config_strings:
        
        ocr_results = find_text_in_crops(rough_crops,options,tesseract_config_string)
        
        text_results = [v['text'] for v in ocr_results.values()]
        assert len(text_results) == 2
            
        # Find datetime
        extracted_datetime = get_datetime_from_strings(text_results,options)
        assert isinstance(extracted_datetime,datetime.datetime) or (extracted_datetime is None)
        if extracted_datetime is not None:        
            assert extracted_datetime.year <= 2023 and extracted_datetime.year >= 1990
    
        if extracted_datetime is not None:
            break
    
    to_return = {}
    to_return['text_results'] = text_results
    to_return['datetime'] = extracted_datetime
    
    if include_crops:
        to_return['ocr_results'] = ocr_results        
    else:
        to_return['ocr_results'] = None
        
    return to_return

# ...def get_datetime_from_image(...)


def try_get_datetime_from_image(filename,include_crops=False,options=None):
    """
    Try/catch wrapper for get_datetime_from_image, defaults to returning 
    metadata only.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    result = {}
    result['error'] = None        
    try:
        result = get_datetime_from_image(filename,include_crops=include_crops,options=options)
    except Exception as e:
        result['error'] = str(e)
    return result


def get_datetimes_for_folder(folder_name,output_file,n_to_sample=-1,options=None):
    """
    Retrieve metadata from every image in [folder_name], and 
    write the results to the .json file [output_file].
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    image_file_names = \
        find_images(folder_name,convert_slashes=True,
                    return_relative_paths=False,recursive=True)
    
    if n_to_sample > 0:
        import random
        random.seed(0)
        image_file_names = random.sample(image_file_names,n_to_sample)
        
    n_cores = 16
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

    return filename_to_results


#%% Interactive driver

if False:
    
    #%% Process images
    
    folder_name = r'g:\temp\island_conservation_camera_traps'
    output_file = r'g:\temp\ocr_results.json'
    n_to_sample = -1
    assert os.path.isdir(folder_name)
    filename_to_results = get_datetimes_for_folder(folder_name,output_file,n_to_sample=n_to_sample)
    

    #%% Explore text
    
    with open(output_file,'r') as f:
        filename_to_results = json.load(f)
    filenames = sorted(list(filename_to_results.keys()))
    print('Loaded results for {} files'.format(len(filename_to_results)))
    
    
    #%%
    
    bad_tokens = ()
    
    # i_fn = 0; fn = filenames[i_fn]
    for i_fn,fn in enumerate(filenames):
        
        image = fn
        results = filename_to_results[fn]
        
        if 'text_results' not in results:
            print('Skipping {}, no results'.format(i_fn))
            continue
        
        s = ' '.join(results['text_results'])
        
        known_bad = False
        for bad_token in bad_tokens:
            if bad_token in s:
                known_bad = True
        if known_bad: 
            continue
                
        extracted_datetime = get_datetime_from_strings([s])
        
        if extracted_datetime is None:
            print('Fallback at {}'.format(i_fn))
            extracted_datetime = get_datetime_from_image(fn)
            
        assert extracted_datetime is not None, 'Error at {}: {}'.format(i_fn,s)
    
        # open_file(fn)
        # get_datetime_from_image(fn)
    
    #%% Write results to an HTML file for testing
          
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


    #%% Other approaches to getting dates from strings
    
    # pip install dateparser
    import dateparser

    # pip install datefinder
    import datefinder

    from dateparser.search import search_dates # noqa

    dateparser_settings = {'PREFER_DATES_FROM':'past','STRICT_PARSING':True}

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
        