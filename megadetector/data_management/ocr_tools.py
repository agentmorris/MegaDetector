"""

ocr_tools.py

Use OCR (via the Tesseract package) to pull metadata (particularly times and
dates from camera trap images).

The general approach is:

* Crop a fixed percentage from the top and bottom of an image, slightly larger
  than the largest examples we've seen of how much space is used for metadata.

* Define the background color as the median pixel value, and find rows that are
  mostly that color to refine the crop.

* Crop to the refined crop, then run pytesseract to extract text.

* Use regular expressions to find time and date.

Prior to using this module:

* Install Tesseract from https://tesseract-ocr.github.io/tessdoc/Installation.html

* pip install pytesseract
   
Known limitations:

* Semi-transparent overlays (which I've only seen on consumer cameras) usually fail.
   
"""

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

from functools import partial
from dateutil.parser import parse as dateparse

import cv2
from PIL import Image, ImageFilter
from tqdm import tqdm

from megadetector.utils.path_utils import find_images
from megadetector.utils.path_utils import open_file
from megadetector.utils import write_html_image_list        
from megadetector.utils.ct_utils import is_iterable
from megadetector.visualization import visualization_utils as vis_utils

# pip install pytesseract
#
# Also install tesseract from: https://github.com/UB-Mannheim/tesseract/wiki, and add
# the installation dir to your path (on Windows, typically C:\Program Files (x86)\Tesseract-OCR)
import pytesseract


#%% Extraction options

class DatetimeExtractionOptions:
    """
    Options used to parameterize datetime extraction in most functions in this module.
    """
    
    def __init__(self):
        
        #: Using a semi-arbitrary metric of how much it feels like we found the 
        #: text-containing region, discard regions that appear to be extraction failures
        self.p_crop_success_threshold = 0.5
        
        #: Pad each crop with a few pixels to make tesseract happy
        self.crop_padding = 10        
        
        #: Discard short text, typically text from the top of the image
        self.min_text_length = 4
        
        #: When we're looking for pixels that match the background color, allow some 
        #: tolerance around the dominant color
        self.background_tolerance = 2
            
        #: We need to see a consistent color in at least this fraction of pixels in our rough 
        #: crop to believe that we actually found a candidate metadata region.
        self.min_background_fraction = 0.3
        
        #: What fraction of the [top,bottom] of the image should we use for our rough crop?
        self.image_crop_fraction = [0.045 , 0.045]
        # self.image_crop_fraction = [0.08 , 0.08]
        
        #: Within that rough crop, how much should we use for determining the background color?
        self.background_crop_fraction_of_rough_crop = 0.5
        
        #: A row is considered a probable metadata row if it contains at least this fraction
        #: of the background color.  This is used only to find the top and bottom of the crop area, 
        #: so it's not that *every* row needs to hit this criteria, only the rows that are generally
        #: above and below the text.
        self.min_background_fraction_for_background_row = 0.5
        
        #: psm 6: "assume a single uniform block of text"
        #: psm 13: raw line
        #: oem: 0 == legacy, 1 == lstm
        #: tesseract_config_string = '--oem 0 --psm 6'
        #:
        #: Try these configuration strings in order until we find a valid datetime
        self.tesseract_config_strings = ['--oem 1 --psm 13','--oem 0 --psm 13',
                                         '--oem 1 --psm 6','--oem 0 --psm 6']
        
        #: If this is False, and one set of options appears to succeed for an image, we'll
        #: stop there.  If this is True, we always run all option sets on every image.
        self.force_all_ocr_options = False
        
        #: Whether to apply PIL's ImageFilter.SHARPEN prior to OCR
        self.apply_sharpening_filter = True
        
        #: Tesseract should be on your system path, but you can also specify the
        #: path explicitly, e.g. you can do either of these:
        #:
        #: * os.environ['PATH'] += r';C:\Program Files\Tesseract-OCR'
        #: * self.tesseract_cmd = 'r"C:\Program Files\Tesseract-OCR\tesseract.exe"'
        self.tesseract_cmd = 'tesseract.exe'


#%% Support functions

def make_rough_crops(image,options=None):
    """
    Crops the top and bottom regions out of an image.
    
    Args:
        image (Image or str): a PIL Image or file name
        options (DatetimeExtractionOptions, optional): OCR parameters
        
    Returns:
        dict: a dict with fields 'top' and 'bottom', each pointing to a new PIL Image        
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
    Given a rough crop from the top or bottom of an image, finds the background color
    and crops to the metadata region.
    
    Within a region of an image (typically a crop from the top-ish or bottom-ish part of 
    an image), tightly crop to the solid portion (typically a region with a black background).

    The success metric is just a binary indicator right now: 1.0 if we found a region we believe
    contains a solid background, 0.0 otherwise.
    
    Args:
        rough_crop (Image): the PIL Image to crop
        crop_location (str): 'top' or 'bottom'
        options (DatetimeExtractionOptions, optional): OCR parameters
        
    Returns:
        tuple: a tuple containing (a cropped_image (Image), p_success (float), padded_image (Image))
    """
    
    if options is None:
        options = DatetimeExtractionOptions()    

    crop_to_solid_region_result = {}
    crop_to_solid_region_result['crop_pil'] = None
    crop_to_solid_region_result['padded_crop_pil'] = None
    crop_to_solid_region_result['p_success'] = 0.0
    
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
    
    if (p_background_value < options.min_background_fraction):
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
    
    # Find the first and last row that are mostly the background color
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
    
    assert (min_y == -1 and max_y == -1) or (min_y != -1 and max_y != -1)
    
    if min_y == -1:
        return crop_to_solid_region_result
    
    if max_y == min_y:
        return crop_to_solid_region_result
    
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
    Finds all text in each Image in the dict [rough_crops]; those images should be pretty small 
    regions by the time they get to this function, roughly the top or bottom 20% of an image.
    
    Args:
        rough_crops (list): list of Image objects that have been cropped close to text
        options (DatetimeExtractionOptions, optional): OCR parameters
        tesseract_config_string (str, optional): optional CLI argument to pass to tesseract.exe
        
    Returns:
        dict: a dict with keys "top" and "bottom", where each value is a dict with keys
        'text' (text found, if any) and 'crop_to_solid_region_results' (metadata about the OCR pass)
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    if tesseract_config_string is None:
        tesseract_config_string = options.tesseract_config_strings[0]
        
    find_text_in_crops_results = {}
    
    # crop_location = 'top'
    # crop_location = 'bottom'
    for crop_location in ('top','bottom'):

        find_text_in_crops_results[crop_location] = {}
        find_text_in_crops_results[crop_location]['text'] = ''
        find_text_in_crops_results[crop_location]['crop_to_solid_region_results'] = None
        
        rough_crop = rough_crops[crop_location]
        
        # Crop to the portion of the rough crop with a solid background color
        crop_to_solid_region_results = crop_to_solid_region(rough_crop,crop_location,options)
        
        find_text_in_crops_results[crop_location]['crop_to_solid_region_results'] = \
            crop_to_solid_region_results
        
        # Try cropping to a solid region; if that doesn't work, try running OCR on the whole
        # rough crop.
        if crop_to_solid_region_results['p_success'] >= options.p_crop_success_threshold:
            padded_crop_pil = crop_to_solid_region_results['padded_crop_pil']
        else:            
            # continue
            padded_crop_pil = rough_crop        
            
        if options.apply_sharpening_filter:
            padded_crop_pil = padded_crop_pil.filter(ImageFilter.SHARPEN)
        
        # Find text in the padded crop
        pytesseract.pytesseract.tesseract_cmd = options.tesseract_cmd
        text = pytesseract.image_to_string(padded_crop_pil, lang='eng', 
                                           config=tesseract_config_string)
        
        text = text.replace('\n', ' ').replace('\r', '').strip()

        find_text_in_crops_results[crop_location]['text'] = text        
                
    # ...for each cropped region
    
    return find_text_in_crops_results
    
# ...def find_text_in_crops(...)
    

def _datetime_string_to_datetime(matched_string):
    """
    Takes an OCR-matched datetime string, does a little cleanup, and parses a date
    from it.
    
    By the time a string gets to this function, it should be a proper date string, with
    no extraneous characters other than spaces around colons or hyphens.
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


def _get_datetime_from_strings(strings,options=None):
    """
    Given a string or list of strings, search for exactly one datetime in those strings.
    using a series of regular expressions.
    
    Strings are currently just concatenated before searching for a datetime.
    """
    
    if options is None:
        options = DatetimeExtractionOptions()    
    
    if isinstance(strings,str):
        s = strings
    else:
        s = ' '.join(strings).lower()
    s = s.replace('—','-')    
    s = ''.join(e for e in s if e.isalnum() or e in ':-/' or e.isspace())
        
    ### AM/PM
    
    # 2013-10-02 11:40:50 AM
    m = re.search('(\d\d\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d)\s+(\d+)\s?:?\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    # 04/01/2017 08:54:00AM
    m = re.search('(\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d\d\d)\s+(\d+)\s?:\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))    
    
    # 2017/04/01 08:54:00AM
    m = re.search('(\d\d\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d)\s+(\d+)\s?:\s?(\d\d)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    # 04/01/2017 08:54AM
    m = re.search('(\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d\d\d)\s+(\d+)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))    
    
    # 2017/04/01 08:54AM
    m = re.search('(\d\d\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d)\s+(\d+)\s?:\s?(\d\d)\s*([a|p]m)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    ### No AM/PM
    
    # 2013-07-27 04:56:35
    m = re.search('(\d\d\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    # 07-27-2013 04:56:35
    m = re.search('(\d\d)\s?-\s?(\d\d)\s?-\s?(\d\d\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    # 2013/07/27 04:56:35
    m = re.search('(\d\d\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    # 07/27/2013 04:56:35
    m = re.search('(\d\d)\s?/\s?(\d\d)\s?/\s?(\d\d\d\d)\s*(\d\d)\s?:\s?(\d\d)\s?:\s?(\d\d)',s)
    if m is not None:        
        return _datetime_string_to_datetime(m.group(0))        
    
    return None
    
# ...def _get_datetime_from_strings(...)


def get_datetime_from_image(image,include_crops=True,options=None):
    """
    Tries to find the datetime string (if present) in an image.
    
    Args:
        image (Image or str): the PIL Image object or image filename in which we should look for
            datetime information.
        include_crops (bool, optional): whether to include cropped images in the return dict (set
            this to False if you're worried about size and you're processing a zillion images)
        options (DatetimeExtractionOptions or list, optional): OCR parameters, either one 
            DatetimeExtractionOptions object or a list of options to try
    
    Returns:
        dict: a dict with fields:
            
            - datetime: Python datetime object, or None
            - text_results: length-2 list of strings
            - all_extracted_datetimes: if we ran multiple option sets, this will contain the 
              datetimes extracted for each option set
            - ocr_results: detailed results from the OCR process, including crops as PIL images;
              only included if include_crops is True
    """
    
    if options is None:
        options = DatetimeExtractionOptions()
    
    if isinstance(image,str):
        image = vis_utils.open_image(image)

    # Crop the top and bottom from the image
    rough_crops = make_rough_crops(image,options)
    assert len(rough_crops) == 2
    
    all_extracted_datetimes = {}
    all_text_results = []
    all_ocr_results = []
    
    extracted_datetime = None
    
    # Find text, possibly trying all config strings
    #
    # tesseract_config_string = options.tesseract_config_strings[0]
    for tesseract_config_string in options.tesseract_config_strings:
        
        ocr_results = find_text_in_crops(rough_crops,options,tesseract_config_string)
        all_ocr_results.append(ocr_results)
        
        text_results = [v['text'] for v in ocr_results.values()]
        assert len(text_results) == 2
        all_text_results.append(text_results)
            
        # Find datetime
        extracted_datetime_this_option_set = _get_datetime_from_strings(text_results,options)
        assert isinstance(extracted_datetime_this_option_set,datetime.datetime) or \
            (extracted_datetime_this_option_set is None)
        
        all_extracted_datetimes[tesseract_config_string] = \
            extracted_datetime_this_option_set
            
        if extracted_datetime_this_option_set is not None:
            if extracted_datetime is None:
                extracted_datetime = extracted_datetime_this_option_set
            if not options.force_all_ocr_options:
                break        
            
    # ...for each set of OCR options
    
    if extracted_datetime is not None:        
        assert extracted_datetime.year <= 2023 and extracted_datetime.year >= 1990

    to_return = {}
    to_return['datetime'] = extracted_datetime
    
    to_return['text_results'] = all_text_results
    to_return['all_extracted_datetimes'] = all_extracted_datetimes
    
    if include_crops:
        to_return['ocr_results'] = all_ocr_results
    else:
        to_return['ocr_results'] = None
        
    return to_return

# ...def get_datetime_from_image(...)


def try_get_datetime_from_image(filename,include_crops=False,options=None):
    """
    Try/catch wrapper for get_datetime_from_image, optionally trying multiple option sets
    until we find a datetime.
    
    Args:
        image (Image or str): the PIL Image object or image filename in which we should look for
            datetime information.
        include_crops (bool, optional): whether to include cropped images in the return dict (set
            this to False if you're worried about size and you're processing a zillion images)
        options (DatetimeExtractionOptions or list, optional): OCR parameters, either one 
            DatetimeExtractionOptions object or a list of options to try
    
    Returns:
        dict: A dict with fields:
            - datetime: Python datetime object, or None
            - text_results: length-2 list of strings
            - all_extracted_datetimes: if we ran multiple option sets, this will contain the 
              datetimes extracted for each option set
            - ocr_results: detailed results from the OCR process, including crops as PIL images;
              only included if include_crops is True
    """
    
    if options is None:
        options = DatetimeExtractionOptions()

    if not is_iterable(options):
        options = [options]
    
    result = {}
    result['error'] = None
    
    for i_option_set,current_options in enumerate(options):
        try:
            result = get_datetime_from_image(filename,include_crops=include_crops,options=current_options)
            result['options_index'] = i_option_set
            if 'datetime' in result and result['datetime'] is not None:
                break
        except Exception as e:
            result['error'] = str(e)
    
    return result


def get_datetimes_for_folder(folder_name,output_file=None,n_to_sample=-1,options=None,
                             n_workers=16,use_threads=False):
    """
    The main entry point for this module.  Tries to retrieve metadata from pixels for every 
    image in [folder_name], optionally the results to the .json file [output_file].
    
    Args:
        folder_name (str): the folder of images to process recursively
        output_file (str, optional): the .json file to which we should write results; if None,
            just returns the results
        n_to_sample (int, optional): for debugging only, used to limit the number of images
            we process
        options (DatetimeExtractionOptions or list, optional): OCR parameters, either one 
            DatetimeExtractionOptions object or a list of options to try for each image
        n_workers (int, optional): the number of parallel workers to use; set to <= 1 to disable
            parallelization
        use_threads (bool, optional): whether to use threads (True) or processes (False) for
            parallelization; not relevant if n_workers <= 1
            
    Returns:
        dict: a dict mapping filenames to datetime extraction results, see try_get_datetime_from_images
        for the format of each value in the dict.
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
            
    if n_workers <= 1:
        
        all_results = []
        for fn_abs in tqdm(image_file_names):
            all_results.append(try_get_datetime_from_image(fn_abs,options=options))
            
    else:    
        
        # Don't spawn more than one worker per image
        if n_workers > len(image_file_names):
            n_workers = len(image_file_names)
            
        if use_threads:
            from multiprocessing.pool import ThreadPool
            pool = ThreadPool(n_workers)
            worker_string = 'threads'        
        else:
            from multiprocessing.pool import Pool
            pool = Pool(n_workers)
            worker_string = 'processes'
            
        print('Starting a pool of {} {}'.format(n_workers,worker_string))
        
        all_results = list(tqdm(pool.imap(
            partial(try_get_datetime_from_image,options=options),image_file_names),
            total=len(image_file_names)))
    
    filename_to_results = {}
    
    # fn_relative = image_file_names[0]
    for i_file,fn_abs in enumerate(image_file_names):
        filename_to_results[fn_abs] = all_results[i_file]
    
    if output_file is not None:
        with open(output_file,'w') as f:
            json.dump(filename_to_results,f,indent=1,default=str)

    return filename_to_results


#%% Interactive driver

if False:
    
    #%% Process images
    
    folder_name = r'g:\temp\island_conservation_camera_traps'
    output_file = r'g:\temp\ocr_results.json'
    from megadetector.utils.path_utils import insert_before_extension
    output_file = insert_before_extension(output_file)
    n_to_sample = -1
    assert os.path.isdir(folder_name)
    options_a = DatetimeExtractionOptions()
    options_b = DatetimeExtractionOptions()
    options_b.image_crop_fraction = [0.08 , 0.08]
    options_a.force_all_ocr_options = False
    options_b.force_all_ocr_options = False
    # all_options = [options_a,options_b]
    all_options = [options_a]
    filename_to_results = get_datetimes_for_folder(folder_name,output_file,
                                                   n_to_sample=n_to_sample,options=all_options)
    

    #%% Load results
    
    # output_file = r"G:\temp\ocr_results.2023.10.31.07.37.54.json"
    with open(output_file,'r') as f:
        filename_to_results = json.load(f)
    filenames = sorted(list(filename_to_results.keys()))
    print('Loaded results for {} files'.format(len(filename_to_results)))
    
    
    #%% Scrap cell
    
    fn = 'g:/camera_traps/camera_trap_images/2018.07.02/newcam/people/DSCF0273.JPG'
    include_crops = False
    options_a = DatetimeExtractionOptions()
    options_b = DatetimeExtractionOptions()
    options_b.image_crop_fraction = [0.08 , 0.08]
    image = vis_utils.open_image(fn) # noqa    
    result = try_get_datetime_from_image(fn,options=[options_a,options_b]) # noqa
    print(result)
    
    # open_file(fn)
    # rough_crops = make_rough_crops(image,options=options)
        
        
    #%% Look for OCR or parsing failures
    
    bad_tokens = ()
    
    files_with_disagreements = set()
            
    # i_fn = 0; fn = filenames[i_fn]
    for i_fn,fn in enumerate(filenames):
        
        image = fn
        results = filename_to_results[fn]
        
        if 'text_results' not in results:
            raise Exception('no results available for {} ({})'.format(i_fn,fn))
            print('Skipping {}, no results'.format(i_fn))
            continue
        
        s = ' '.join([x[0] for x in results['text_results']])
        
        known_bad = False
        for bad_token in bad_tokens:
            if bad_token in s:
                known_bad = True
        if known_bad: 
            continue
                
        extracted_datetime = results['datetime']
        
        # If we have a datetime, make sure all successful OCR results agree
        if extracted_datetime is not None:
            for config_string in results['all_extracted_datetimes']:
                if results['all_extracted_datetimes'][config_string] is not None:
                    if results['all_extracted_datetimes'][config_string] != extracted_datetime:
                        files_with_disagreements.add(fn)
        else:
            print('Falling back for {} ({})'.format(i_fn,fn))
            ocr_results = get_datetime_from_image(fn)
            extracted_datetime = ocr_results['datetime']
        
        if extracted_datetime is None:
            print('Failure at {}: {}'.format(i_fn,s))
        
        # open_file(fn)
        # get_datetime_from_image(fn)
    
    
    #%% Write results to an HTML file for testing
          
    n_to_sample = 5000
    if (n_to_sample >= 0) and (len(filename_to_results) > n_to_sample):
        filenames = sorted(list(filename_to_results.keys()))        
        import random
        random.seed(0)
        keys = random.sample(filenames,n_to_sample)
        filename_to_results = {k: filename_to_results[k] for k in keys}

    preview_dir = r'g:\temp\ocr-preview'
    os.makedirs(preview_dir,exist_ok=True)
    
    def resize_image_for_preview(fn_abs):
        fn_relative = os.path.relpath(fn_abs,folder_name)        
        resized_image = vis_utils.resize_image(fn_abs,target_width=600)
        resized_fn = os.path.join(preview_dir,fn_relative)
        os.makedirs(os.path.dirname(resized_fn),exist_ok=True)
        resized_image.save(resized_fn)
        return resized_fn
        
    # Resize images in parallel
    n_rendering_workers = 16
        
    if n_rendering_workers <= 1:
        for fn_abs in tqdm(filename_to_results.keys()):
            resize_image_for_preview(fn_abs)
    else:
        # from multiprocessing.pool import Pool as RenderingPool; worker_string = 'processes'
        from multiprocessing.pool import ThreadPool as RenderingPool; worker_string = 'threads'
        pool = RenderingPool(n_rendering_workers)
        
        print('Starting rendering pool with {} {}'.format(n_rendering_workers,worker_string))
    
        _ = list(tqdm(pool.imap(resize_image_for_preview,filename_to_results.keys()),
                      total=len(filename_to_results)))
    
    
    def make_datetime_preview_page(filenames,html_file):
        
        html_image_list = []
        html_options = write_html_image_list.write_html_image_list()
        html_options['maxFiguresPerHtmlFile'] = 2500
        html_options['defaultImageStyle'] = 'margin:0px;margin-top:5px;margin-bottom:30px;'
        
        # fn_abs = filenames[0]
        for fn_abs in filenames:
            
            fn_relative = os.path.relpath(fn_abs,folder_name)        
            # resized_fn = os.path.join(preview_dir,fn_relative)
            results_this_image = filename_to_results[fn_abs]
                
            extracted_datetime = results_this_image['datetime']
            title = 'Image: {}<br/>Extracted datetime: {}'.format(fn_relative,extracted_datetime)
            html_image_list.append({'filename':fn_relative,'title':title})
            
            # ...for each crop
        
        # ...for each image
                
        html_options['makeRelative'] = True
        write_html_image_list.write_html_image_list(html_file,
                                                    html_image_list,
                                                    html_options)
        open_file(html_file)
        return html_image_list
    
    failed_files = []
    for fn_abs in filename_to_results:
        results_this_image = filename_to_results[fn_abs]
        if results_this_image['datetime'] is None:
            failed_files.append(fn_abs)
    
    print('Found {} failures'.format(len(failed_files)))
    
    output_summary_file = os.path.join(preview_dir,'summary.html')
    html_image_list = make_datetime_preview_page(sorted(list(filename_to_results.keys())),output_summary_file)
    
    failure_summary_file = os.path.join(preview_dir,'failures.html')    
    html_image_list_failures = make_datetime_preview_page(failed_files,failure_summary_file)
    
    filenames = failed_files
    html_file = failure_summary_file

    
    #%% Other approaches to getting dates from strings
    
    # ...that didn't really work out.
    
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

