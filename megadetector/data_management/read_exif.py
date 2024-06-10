"""

read_exif.py

Given a folder of images, reads relevant metadata (EXIF/IPTC/XMP) fields from all images, 
and writes them to  a .json or .csv file.  

This module can use either PIL (which can only reliably read EXIF data) or exiftool (which
can read everything).  The latter approach expects that exiftool is available on the system
path.  No attempt is made to be consistent in format across the two approaches.

"""

#%% Imports and constants

import os
import subprocess
import json
from datetime import date, datetime

from multiprocessing.pool import ThreadPool as ThreadPool
from multiprocessing.pool import Pool as Pool

from tqdm import tqdm
from PIL import Image, ExifTags

from megadetector.utils.path_utils import find_images, is_executable
from megadetector.utils.ct_utils import args_to_object
from megadetector.utils.ct_utils import image_file_to_camera_folder

debug_max_images = None


#%% Options

class ReadExifOptions:
    """
    Parameters controlling metadata extraction.
    """
    
    def __init__(self):
        
        #: Enable additional debug console output
        self.verbose = False
        
        #: If this is True and an output file is specified for read_exif_from_folder,
        #: and we encounter a serialization issue, we'll return the results but won't
        #: error.    
        self.allow_write_error = False
        
        #: Number of concurrent workers, set to <= 1 to disable parallelization
        self.n_workers = 1
        
        #: Should we use threads (vs. processes) for parallelization?
        #:
        #: Not relevant if n_workers is <= 1.
        self.use_threads = True
            
        #: "File" and "ExifTool" are tag types used by ExifTool to report data that 
        #: doesn't come from EXIF, rather from the file (e.g. file size).
        self.tag_types_to_ignore = set(['File','ExifTool'])
        
        #: Include/exclude specific tags (tags_to_include and tags_to_exclude are mutually incompatible)
        #:
        #: A useful set of tags one might want to limit queries for:
        #:
        #: options.tags_to_include = ['DateTime','Model','Make','ExifImageWidth','ExifImageHeight',
        #: 'DateTimeOriginal','Orientation']    
        self.tags_to_include = None
        
        #: Include/exclude specific tags (tags_to_include and tags_to_exclude are mutually incompatible)
        self.tags_to_exclude = None
        
        #: The command line to invoke if using exiftool, can be an absolute path to exiftool.exe, or
        #: can be just "exiftool", in which case it should be on your system path.
        self.exiftool_command_name = 'exiftool'
        
        #: How should we handle byte-formatted EXIF tags?
        #:
        #: 'convert_to_string': convert to a Python string
        #: 'delete': don't include at all
        #: 'raw': include as a byte string
        self.byte_handling = 'convert_to_string' # 'convert_to_string','delete','raw'
        
        #: Should we use exiftool or PIL?
        self.processing_library = 'pil' # 'exiftool','pil'
        

class ExifResultsToCCTOptions:
    """
    Options controlling the behavior of exif_results_to_cct() (which reformats the datetime information)
    extracted by read_exif_from_folder().
    """
    
    def __init__(self):
        
        #: Timestamps older than this are assumed to be junk; lots of cameras use a
        #: default time in 2000.
        self.min_valid_timestamp_year = 2001
    
        #: The EXIF tag from which to pull datetime information
        self.exif_datetime_tag = 'DateTimeOriginal'
    
        #: Function for extracting location information, should take a string
        #: and return a string.  Defaults to ct_utils.image_file_to_camera_folder.  If
        #: this is None, location is written as "unknown".
        self.filename_to_location_function = image_file_to_camera_folder
    

#%% Functions

def _get_exif_ifd(exif):
    """
    Read EXIF data from by finding the EXIF offset and reading tags directly
    
    https://github.com/python-pillow/Pillow/issues/5863
    """
    
    # Find the offset for all the EXIF information
    for key, value in ExifTags.TAGS.items():
        if value == "ExifOffset":
            break
    info = exif.get_ifd(key)
    return {
        ExifTags.TAGS.get(key, key): value
        for key, value in info.items()
    }


def read_pil_exif(im,options=None):
    """
    Read all the EXIF data we know how to read from an image, using PIL.  This is primarily
    an internal function; the main entry point for single-image EXIF information is 
    read_exif_tags_for_image().
    
    Args:
        im (str or PIL.Image.Image): image (as a filename or an Image object) from which 
            we should read EXIF data.
    
    Returns:
        dict: a dictionary mapping EXIF tag names to their values
    """
    
    if options is None:
        options = ReadExifOptions()
        
    image_name = '[image]'
    if isinstance(im,str):
        image_name = im
        im = Image.open(im)
        
    exif_tags = {}
    try:
        exif_info = im.getexif()
    except Exception:
        exif_info = None
        
    if exif_info is None:
        return exif_tags
    
    for k, v in exif_info.items():
        assert isinstance(k,str) or isinstance(k,int), \
            'Invalid EXIF key {}'.format(str(k))
        if k in ExifTags.TAGS:
            exif_tags[ExifTags.TAGS[k]] = str(v)
        else:
            # print('Warning: unrecognized EXIF tag: {}'.format(k))
            exif_tags[k] = str(v)
    
    exif_ifd_tags = _get_exif_ifd(exif_info)
    
    for k in exif_ifd_tags.keys():
        v = exif_ifd_tags[k]
        if k in exif_tags:
            if options.verbose:
                print('Warning: redundant EXIF values for {} in {}:\n{}\n{}'.format(
                    k,image_name,exif_tags[k],v))
        else:
            exif_tags[k] = v
    
    exif_tag_names = list(exif_tags.keys())
    
    # Type conversion and cleanup
    # 
    # Most quirky types will get serialized to string when we write to .json.
    for k in exif_tag_names:
        
        if isinstance(exif_tags[k],bytes):
            
            if options.byte_handling == 'delete':
                del exif_tags[k]
            elif options.byte_handling == 'raw':
                pass
            else:
                assert options.byte_handling == 'convert_to_string'
                exif_tags[k] = str(exif_tags[k])
        
        elif isinstance(exif_tags[k],str):
            
            exif_tags[k] = exif_tags[k].strip()
            
    return exif_tags

# ...read_pil_exif()


def format_datetime_as_exif_datetime_string(dt):
    """
    Returns a Python datetime object rendered using the standard EXIF datetime
    string format ('%Y:%m:%d %H:%M:%S')
    """
    
    return datetime.strftime(dt, '%Y:%m:%d %H:%M:%S')
    

def parse_exif_datetime_string(s,verbose=False):
    """"
    Exif datetimes are strings, but in a standard format:        
        
    %Y:%m:%d %H:%M:%S
    
    Parses one of those strings into a Python datetime object.
    
    Args:
        s (str): datetime string to parse, should be in standard EXIF datetime format
        verbose (bool, optional): enable additional debug output
    
    Returns:
        datetime: the datetime object created from [s]
    """
    
    dt = None
    try:
        dt = datetime.strptime(s, '%Y:%m:%d %H:%M:%S')
    except Exception:
        if verbose:
            print('Warning: could not parse datetime {}'.format(str(s)))
    return dt


def _filter_tags(tags,options):
    """
    Internal function used to include/exclude specific tags from the exif_tags
    dict.
    """
    
    if options is None:
        return tags
    if options.tags_to_include is None and options.tags_to_exclude is None:
        return tags
    if options.tags_to_include is not None:
        assert options.tags_to_exclude is None, "tags_to_include and tags_to_exclude are incompatible"
        tags_to_return = {}
        for tag_name in tags.keys():
            if tag_name in options.tags_to_include:
                tags_to_return[tag_name] = tags[tag_name]
        return tags_to_return
    if options.tags_to_exclude is not None:
        assert options.tags_to_include is None, "tags_to_include and tags_to_exclude are incompatible"
        tags_to_return = {}
        for tag_name in tags.keys():
            if tag_name not in options.tags_to_exclude:
                tags_to_return[tag_name] = tags[tag_name]
        return tags_to_return


def read_exif_tags_for_image(file_path,options=None):
    """
    Get relevant fields from EXIF data for an image
    
    Returns:
        dict: a dict with fields 'status' (str) and 'tags'. The exact format of 'tags' depends on 
        options (ReadExifOptions, optional): parameters controlling metadata extraction
        options.processing_library:
            
            - For exiftool, 'tags' is a list of lists, where each element is (type/tag/value)
            - For PIL, 'tags' is a dict (str:str)
    """
    
    if options is None:
        options = ReadExifOptions()
    
    result = {'status':'unknown','tags':[]}
    
    if options.processing_library == 'pil':
        
        try:
            exif_tags = read_pil_exif(file_path,options)

        except Exception as e:
            if options.verbose:
                print('Read failure for image {}: {}'.format(
                    file_path,str(e)))
            result['status'] = 'read_failure'
            result['error'] = str(e)
        
        if result['status'] == 'unknown':
            if exif_tags is None:            
                result['status'] = 'empty_read'
            else:
                result['status'] = 'success'
                result['tags'] = _filter_tags(exif_tags,options)
                            
        return result
        
    elif options.processing_library == 'exiftool':
        
        # -G means "Print group name for each tag", e.g. print:
        #
        # [File]          Bits Per Sample                 : 8
        #
        # ...instead of:
        #
        # Bits Per Sample                 : 8
        proc = subprocess.Popen([options.exiftool_command_name, '-G', file_path],
                                stdout=subprocess.PIPE, encoding='utf8')
        
        exif_lines = proc.stdout.readlines()    
        exif_lines = [s.strip() for s in exif_lines]
        if ( (exif_lines is None) or (len(exif_lines) == 0) or not \
            any([s.lower().startswith('[exif]') for s in exif_lines])):
            result['status'] = 'failure'
            return result
        
        # A list of three-element lists (type/tag/value)
        exif_tags = []
        
        # line_raw = exif_lines[0]
        for line_raw in exif_lines:
            
            # A typical line:
            #
            # [ExifTool]      ExifTool Version Number         : 12.13
            
            line = line_raw.strip()
            
            # Split on the first occurrence of ":"
            tokens = line.split(':',1)
            assert(len(tokens) == 2), 'EXIF tokenization failure ({} tokens, expected 2)'.format(
                len(tokens))
            
            field_value = tokens[1].strip()        
            
            field_name_type = tokens[0].strip()        
            field_name_type_tokens = field_name_type.split(None,1)
            assert len(field_name_type_tokens) == 2, 'EXIF tokenization failure'
            
            field_type = field_name_type_tokens[0].strip()
            assert field_type.startswith('[') and field_type.endswith(']'), \
                'Invalid EXIF field {}'.format(field_type)
            field_type = field_type[1:-1]
            
            if field_type in options.tag_types_to_ignore:
                if options.verbose:
                    print('Ignoring tag with type {}'.format(field_type))
                continue        
            
            field_name = field_name_type_tokens[1].strip()
            if options.tags_to_exclude is not None and field_name in options.tags_to_exclude:
                continue
            if options.tags_to_include is not None and field_name not in options.tags_to_include:
                continue
            tag = [field_type,field_name,field_value]
            
            exif_tags.append(tag)
            
        # ...for each output line
            
        result['status'] = 'success'
        result['tags'] = exif_tags
        return result
    
    else:
        
        raise ValueError('Unknown processing library {}'.format(
            options.processing_library))

    # ...which processing library are we using?
    
# ...read_exif_tags_for_image()


def _populate_exif_data(im, image_base, options=None):
    """
    Populate EXIF data into the 'exif_tags' field in the image object [im].
    
    im['file_name'] should be prepopulated, relative to image_base.
    
    Returns a modified version of [im], also modifies [im] in place.
    """
    
    if options is None:
        options = ReadExifOptions()

    fn = im['file_name']
    if options.verbose:
        print('Processing {}'.format(fn))
    
    try:
        
        file_path = os.path.join(image_base,fn)
        assert os.path.isfile(file_path), 'Could not find file {}'.format(file_path)
        result = read_exif_tags_for_image(file_path,options)
        if result['status'] == 'success':
            exif_tags = result['tags']            
            im['exif_tags'] = exif_tags
        else:
            im['exif_tags'] = None
            im['status'] = result['status']
            if 'error' in result:
                im['error'] = result['error']
            if options.verbose:
                print('Error reading EXIF data for {}'.format(file_path))
    
    except Exception as e:
        
        s = 'Error on {}: {}'.format(fn,str(e))
        print(s)
        im['error'] = s
        im['status'] = 'read failure'
        im['exif_tags'] = None
    
    return im

# ..._populate_exif_data()


def _create_image_objects(image_files,recursive=True):
    """
    Create empty image objects for every image in [image_files], which can be a 
    list of relative paths (which will get stored without processing, so the base 
    path doesn't matter here), or a folder name.
    
    Returns a list of dicts with field 'file_name' (a relative path).
    
    "recursive" is ignored if "image_files" is a list.
    """
    
    # Enumerate *relative* paths
    if isinstance(image_files,str):    
        print('Enumerating image files in {}'.format(image_files))
        assert os.path.isdir(image_files), 'Invalid image folder {}'.format(image_files)
        image_files = find_images(image_files,
                                  recursive=recursive,
                                  return_relative_paths=True,
                                  convert_slashes=True)
        
    images = []
    for fn in image_files:
        im = {}
        im['file_name'] = fn
        images.append(im)
    
    if debug_max_images is not None:
        print('Trimming input list to {} images'.format(debug_max_images))
        images = images[0:debug_max_images]
    
    return images


def _populate_exif_for_images(image_base,images,options=None):
    """
    Main worker loop: read EXIF data for each image object in [images] and 
    populate the image objects in place.
    
    'images' should be a list of dicts with the field 'file_name' containing
    a relative path (relative to 'image_base').    
    """
    
    if options is None:
        options = ReadExifOptions()

    if options.n_workers == 1:
      
        results = []
        for im in tqdm(images):
            results.append(_populate_exif_data(im,image_base,options))
        
    else:
        
        from functools import partial
        if options.use_threads:
            print('Starting parallel thread pool with {} workers'.format(options.n_workers))
            pool = ThreadPool(options.n_workers)
        else:
            print('Starting parallel process pool with {} workers'.format(options.n_workers))
            pool = Pool(options.n_workers)
    
        results = list(tqdm(pool.imap(partial(_populate_exif_data,image_base=image_base,
                                        options=options),images),total=len(images)))

    return results


def _write_exif_results(results,output_file):
    """
    Write EXIF information to [output_file].
    
    'results' is a list of dicts with fields 'exif_tags' and 'file_name'.

    Writes to .csv or .json depending on the extension of 'output_file'.         
    """
    
    if output_file.endswith('.json'):
        
        with open(output_file,'w') as f:
            json.dump(results,f,indent=1,default=str)
            
    elif output_file.endswith('.csv'):
        
        # Find all EXIF tags that exist in any image
        all_keys = set()
        for im in results:
            
            keys_this_image = set()
            exif_tags = im['exif_tags']
            file_name = im['file_name']
            for tag in exif_tags:
                tag_name = tag[1]
                assert tag_name not in keys_this_image, \
                    'Error: tag {} appears twice in image {}'.format(
                        tag_name,file_name)
                all_keys.add(tag_name)
                
            # ...for each tag in this image
            
        # ...for each image
        
        all_keys = sorted(list(all_keys))
        
        header = ['File Name']
        header.extend(all_keys)
        
        import csv
        with open(output_file,'w') as csvfile:
            
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(header)
            
            for im in results:
                
                row = [im['file_name']]
                kvp_this_image = {tag[1]:tag[2] for tag in im['exif_tags']}
                
                for i_key,key in enumerate(all_keys):
                    value = ''
                    if key in kvp_this_image:
                        value = kvp_this_image[key]
                    row.append(value)                                        
                # ...for each key that *might* be present in this image
                
                assert len(row) == len(header)
                
                writer.writerow(row)
                
            # ...for each image
            
        # ...with open()
    
    else:
        
        raise ValueError('Could not determine output type from file {}'.format(
            output_file))
        
    # ...if we're writing to .json/.csv
    
    print('Wrote results to {}'.format(output_file))

# ..._write_exif_results(...)


def read_exif_from_folder(input_folder,output_file=None,options=None,filenames=None,recursive=True):
    """
    Read EXIF data for a folder of images.
    
    Args:
        input_folder (str): folder to process; if this is None, [filenames] should be a list of absolute
            paths
        output_file (str, optional): .json file to which we should write results; if this is None, results
            are returned but not written to disk
        options (ReadExifOptions, optional): parameters controlling metadata extraction
        filenames (list, optional): allowlist of relative filenames (if [input_folder] is not None) or
            a list of absolute filenames (if [input_folder] is None)
        recursive (bool, optional): whether to recurse into [input_folder], not relevant if [input_folder]
            is None.
        verbose (bool, optional): enable additional debug output
            
    Returns:
        list: list of dicts, each of which contains EXIF information for one images.  Fields include at least:
            * 'file_name': the relative path to the image
            * 'exif_tags': a dict of EXIF tags whose exact format depends on [options.processing_library].
    """
    
    if options is None:
        options = ReadExifOptions()
    
    # Validate options
    if options.tags_to_include is not None:
        assert options.tags_to_exclude is None, "tags_to_include and tags_to_exclude are incompatible"
    if options.tags_to_exclude is not None:
        assert options.tags_to_include is None, "tags_to_include and tags_to_exclude are incompatible"    
    
    if input_folder is None:
        input_folder = ''
    if len(input_folder) > 0:
        assert os.path.isdir(input_folder), \
            '{} is not a valid folder'.format(input_folder)

    assert (len(input_folder) > 0) or (filenames is not None), \
        'Must specify either a folder or a list of files'
        
    if output_file is not None:    
        
        assert output_file.lower().endswith('.json') or output_file.lower().endswith('.csv'), \
            'I only know how to write results to .json or .csv'
            
        try:
            with open(output_file, 'a') as f:
                if not f.writable():
                    raise IOError('File not writable')
        except Exception:
            print('Could not write to file {}'.format(output_file))
            raise
        
    if options.processing_library == 'exif':
        assert is_executable(options.exiftool_command_name), 'exiftool not available'

    if filenames is None:
        images = _create_image_objects(input_folder,recursive=recursive)
    else:
        assert isinstance(filenames,list)
        images = _create_image_objects(filenames)
        
    results = _populate_exif_for_images(input_folder,images,options)
    
    if output_file is not None:
        try:
            _write_exif_results(results,output_file)
        except Exception as e:
            if not options.allow_write_error:
                raise
            else:
                print('Warning: error serializing EXIF data: {}'.format(str(e)))                
        
    return results

# ...read_exif_from_folder(...)


def exif_results_to_cct(exif_results,cct_output_file=None,options=None):
    """
    Given the EXIF results for a folder of images read via read_exif_from_folder,
    create a COCO Camera Traps .json file that has no annotations, but 
    attaches image filenames to locations and datetimes.
    
    Args:
        exif_results (str or list): the filename (or loaded list) containing the results
          from read_exif_from_folder
        cct_file (str,optional): the filename to which we should write COCO-Camera-Traps-formatted
          data
          
    Returns:
        dict: a COCO Camera Traps dict (with no annotations).
    """
    
    if options is None:
        options = ExifResultsToCCTOptions()
        
    if isinstance(exif_results,str):
        print('Reading EXIF results from {}'.format(exif_results))
        with open(exif_results,'r') as f:
            exif_results = json.load(f)
    else:
        assert isinstance(exif_results,list)
            
    now = datetime.now()

    image_info = []

    images_without_datetime = []
    images_with_invalid_datetime = []
    
    # exif_result = exif_results[0]
    for exif_result in tqdm(exif_results):
        
        im = {}
        
        # By default we assume that each leaf-node folder is a location
        if options.filename_to_location_function is None:
            im['location'] = 'unknown'
        else:
            im['location'] = options.filename_to_location_function(exif_result['file_name'])            
            
        im['file_name'] = exif_result['file_name']
        im['id'] = im['file_name']
        
        if ('exif_tags' not in exif_result) or (exif_result['exif_tags'] is None) or \
            (options.exif_datetime_tag not in exif_result['exif_tags']): 
            exif_dt = None
        else:
            exif_dt = exif_result['exif_tags'][options.exif_datetime_tag]
            exif_dt = parse_exif_datetime_string(exif_dt)
        if exif_dt is None:
            im['datetime'] = None
            images_without_datetime.append(im['file_name'])
        else:
            dt = exif_dt
            
            # An image from the future (or within the last 24 hours) is invalid
            if (now - dt).total_seconds() <= 1*24*60*60:
                print('Warning: datetime for {} is {}'.format(
                    im['file_name'],dt))
                im['datetime'] = None            
                images_with_invalid_datetime.append(im['file_name'])
            
            # An image from before the dawn of time is also invalid
            elif dt.year < options.min_valid_timestamp_year:
                print('Warning: datetime for {} is {}'.format(
                    im['file_name'],dt))
                im['datetime'] = None
                images_with_invalid_datetime.append(im['file_name'])
            
            else:
                im['datetime'] = dt

        image_info.append(im)
        
    # ...for each exif image result

    print('Parsed EXIF datetime information, unable to parse EXIF date from {} of {} images'.format(
        len(images_without_datetime),len(exif_results)))

    d = {}
    d['info'] = {}
    d['images'] = image_info
    d['annotations'] = []
    d['categories'] = []
    
    def json_serialize_datetime(obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        raise TypeError('Object {} (type {}) not serializable'.format(
            str(obj),type(obj)))
        
    if cct_output_file is not None:
        with open(cct_output_file,'w') as f:
            json.dump(d,f,indent=1,default=json_serialize_datetime)
    
    return d

# ...exif_results_to_cct(...)

    
#%% Interactive driver

if False:
    
    #%%
    
    input_folder = r'C:\temp\md-name-testing'
    output_file = None # r'C:\temp\md-name-testing\exif.json'
    options = ReadExifOptions()
    options.verbose = False
    options.n_workers = 10
    options.use_threads = False
    options.processing_library = 'pil'
    # options.processing_library = 'exiftool'
    options.tags_to_include = ['DateTime','Model','Make','ExifImageWidth','ExifImageHeight','DateTime','DateTimeOriginal','Orientation']
    # options.tags_to_exclude = ['MakerNote']
    
    results = read_exif_from_folder(input_folder,output_file,options)

    #%%
    
    with open(output_file,'r') as f:
        d = json.load(f)
        

#%% Command-line driver

import argparse
import sys

def main():

    options = ReadExifOptions()
    
    parser = argparse.ArgumentParser(description=('Read EXIF information from all images in' + \
                                                  ' a folder, and write the results to .csv or .json'))

    parser.add_argument('input_folder', type=str, 
                        help='Folder of images from which we should read EXIF information')
    parser.add_argument('output_file', type=str,
                        help='Output file (.json) to which we should write EXIF information')
    parser.add_argument('--n_workers', type=int, default=1,
                        help='Number of concurrent workers to use (defaults to 1)')
    parser.add_argument('--use_threads', action='store_true',
                        help='Use threads (instead of processes) for multitasking')
    parser.add_argument('--processing_library', type=str, default=options.processing_library,
                        help='Processing library (exif or pil)')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()    
    args_to_object(args, options)
    options.processing_library = options.processing_library.lower()
    
    read_exif_from_folder(args.input_folder,args.output_file,options)
    
if __name__ == '__main__':
    main()
