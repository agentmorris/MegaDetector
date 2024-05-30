"""

rename_images.py.py

Copies images from a possibly-nested folder structure to a flat folder structure, including EXIF
timestamps in each filename.  Loosely equivalent to camtrapR's imageRename() function.

"""

#%% Imports and constants

import os

from megadetector.utils.path_utils import \
    find_images, insert_before_extension, parallel_copy_files
from megadetector.data_management.read_exif import \
    ReadExifOptions, read_exif_from_folder


#%% Functions

def rename_images(input_folder,
                  output_folder,
                  dry_run=False,
                  verbose=False,
                  read_exif_options=None,
                  n_copy_workers=8):
    """
    For the given image struct in COCO format and associated list of annotations, reformats the 
    detections into labelme format.  
    
    Args:
        input_folder: the folder to search for images, always recursive
        output_folder: the folder to which we will copy images; cannot be the
            same as [input_folder]
        dry_run: only map images, don't actually copy
        verbose (bool, optional): enable additional debug output
        read_exif_options (ReadExifOptions, optional): parameters controlling the reading of
            EXIF information
        n_copy_workers (int, optional): number of parallel threads to use for copying
                    
    Returns:
        dict: a dict mapping relative filenames in the input folder to relative filenames in the output 
        folder
    """
    
    assert os.path.isdir(input_folder), 'Input folder {} does not exist'.format(
        input_folder)
    
    if not dry_run:
        os.makedirs(output_folder,exist_ok=True)
    
    # Read exif information
    if read_exif_options is None:
        read_exif_options = ReadExifOptions()
   
    read_exif_options.tags_to_include = ['DateTime','Model','Make','ExifImageWidth','ExifImageHeight','DateTime',
                                         'DateTimeOriginal']    
    read_exif_options.verbose = False
    
    exif_info = read_exif_from_folder(input_folder=input_folder,
                                      output_file=None,
                                      options=read_exif_options,
                                      filenames=None,recursive=True)
    
    print('Read EXIF information for {} images'.format(len(exif_info)))
    
    filename_to_exif_info = {info['file_name']:info for info in exif_info}
    
    image_files = find_images(input_folder,return_relative_paths=True,convert_slashes=True,recursive=True)
    
    for fn in image_files:
        assert fn in filename_to_exif_info, 'No EXIF info available for {}'.format(fn)
    
    input_fn_relative_to_output_fn_relative = {}
    
    # fn_relative = image_files[0]
    for fn_relative in image_files:
        
        input_fn_abs = os.path.join(input_folder,fn_relative)
        image_exif_info = filename_to_exif_info[fn_relative]
        if 'exif_tags' in image_exif_info:
            image_exif_info = image_exif_info['exif_tags']
        
        if image_exif_info is None or \
            'DateTimeOriginal' not in image_exif_info or \
            image_exif_info['DateTimeOriginal'] is None:
                
            dt_tag = 'unknown_datetime'
            print('Warning: no datetime for {}'.format(fn_relative))
            
        else:
            
            dt_tag = str(image_exif_info['DateTimeOriginal']).replace(':','-').replace(' ','_').strip()            
        
        flat_filename = fn_relative.replace('\\','/').replace('/','_')
        
        output_fn_relative = insert_before_extension(flat_filename,dt_tag)
        
        input_fn_relative_to_output_fn_relative[fn_relative] = output_fn_relative
        
    if not dry_run:
        
        input_fn_abs_to_output_fn_abs = {}
        for input_fn_relative in input_fn_relative_to_output_fn_relative:
            output_fn_relative = input_fn_relative_to_output_fn_relative[input_fn_relative]
            input_fn_abs = os.path.join(input_folder,input_fn_relative)
            output_fn_abs = os.path.join(output_folder,output_fn_relative)
            input_fn_abs_to_output_fn_abs[input_fn_abs] = output_fn_abs
            
            parallel_copy_files(input_file_to_output_file=input_fn_abs_to_output_fn_abs,
                                max_workers=n_copy_workers, 
                                use_threads=True, 
                                overwrite=True, 
                                verbose=verbose)
            
    return input_fn_relative_to_output_fn_relative

# ...def rename_images()


#%% Interactive driver

if False:
    
    pass

    #%% Configure options
    
    input_folder = r'G:\camera_traps\camera_trap_videos\2024.05.25\cam3'
    output_folder = r'G:\camera_traps\camera_trap_videos\2024.05.25\cam3_flat'
    dry_run = False
    verbose = True
    read_exif_options = ReadExifOptions()
    read_exif_options.tags_to_include = ['DateTime','Model','Make','ExifImageWidth','ExifImageHeight','DateTime',
                               'DateTimeOriginal']    
    read_exif_options.n_workers = 8
    read_exif_options.verbose = verbose    
    n_copy_workers = 8
    
    
    #%% Programmatic execution
    
    input_fn_relative_to_output_fn_relative = rename_images(input_folder,
                                                            output_folder,
                                                            dry_run=dry_run,
                                                            verbose=verbose,
                                                            read_exif_options=read_exif_options,
                                                            n_copy_workers=n_copy_workers)
    

#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='Copies images from a possibly-nested folder structure to a flat folder structure, ' + \
            'adding datetime information from EXIF to each filename')
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='The folder to search for images, always recursive')
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='The folder to which we should write the flattened image structure')
    
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help="Only map images, don't actually copy")

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    rename_images(args.input_folder,args.output_folder,dry_run=args.dry_run,
                  verbose=True,read_exif_options=None)

if __name__ == '__main__':
    main()
