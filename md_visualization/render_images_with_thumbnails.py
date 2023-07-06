########
# 
# render_images_with_thumbnails.py
# 
# Render an output image with one primary and crops from many secondary images,
# used to check whether suspicious detections are actually false positives or not.
#
########

#%% Constants

import math
import os
import random

from md_visualization import visualization_utils as vis_utils
from PIL import Image


#%% Support functions

def crop_image_with_normalized_coordinates(
        image,
        bounding_box):
    """
    Args:
        image: image to crop
        bounding_box: tuple formatted as (x,y,w,h), where (0,0) is the
        upper-left of the image, and coordinates are normalized
        (so (0,0,1,1) is a box containing the entire image).
    """
    
    im_width, im_height = image.size
    (x_norm, y_norm, w_norm, h_norm) = bounding_box
    (x, y, w, h) = (x_norm * im_width,
                    y_norm * im_height,
                    w_norm * im_width,
                    h_norm * im_height)
    return image.crop((x, y, x+w, y+h))



#%% Main function

def render_images_with_thumbnails(
        primary_image_filename,
        primary_image_width,
        secondary_image_filename_list,
        secondary_image_bounding_box_list,
        cropped_grid_width,
        output_image_filename,
        primary_image_location='right'):
    """
    Given a primary image filename and a list of secondary images, writes to
    the provided output_image_filename an image where the one
    side is the primary image, and the other side is a grid of the 
    secondary images, cropped according to the provided list of bounding
    boxes.

    The output file will be primary_image_width + cropped_grid_width pixels
    wide.

    The height of the output image will be determined by the original aspect
    ratio of the primary image. 
    
    Args:
        primary_image_filename: filename of the primary image to load as str
        primary_image_width: width at which to render the primary image; if this is 
            None, will render at the original image width.
        secondary_image_filename_list: list of strs that are the filenames of
            the secondary images.
        secondary_image_bounding_box_list: list of tuples, one per secondary
            image. Each tuple is a bounding box of the secondary image,
            formatted as (x,y,w,h), where (0,0) is the upper-left of the image,
            and coordinates are normalized (so (0,0,1,1) is a box containing
            the entire image.
        cropped_grid_width: width of all the cropped images
        output_image_filename: str of the filename to write the output image        
        primary_image_location: 'right' or left'; reserving 'top', 'bottom', etc.
            for future use.
    """

    # Check to make sure the arguments are reasonable
    assert(len(secondary_image_filename_list) ==
           len(secondary_image_bounding_box_list)), \
           'Length of secondary image list and bounding box list should be equal'

    assert primary_image_location in ['left','right']
    
    # Load primary image and resize to desired width
    primary_image = vis_utils.load_image(primary_image_filename)
    if primary_image_width is not None:
        primary_image = vis_utils.resize_image(primary_image, primary_image_width, 
                                               target_height=-1)

    # Compute the number of grid elements for the secondary images
    # to best fit the available aspect ratio
    grid_width = cropped_grid_width
    grid_height = primary_image.size[1]
    grid_aspect = grid_width / grid_height
    
    sample_crop_width = secondary_image_bounding_box_list[0][2]
    sample_crop_height = secondary_image_bounding_box_list[0][3]
    
    n_crops = len(secondary_image_filename_list)
    
    optimal_n_rows = None
    optimal_aspect_error = None
    
    for candidate_n_rows in range(1,n_crops+1):
        candidate_n_cols = math.ceil(n_crops / candidate_n_rows)
        candidate_grid_aspect = (candidate_n_cols*sample_crop_width) / \
          (candidate_n_rows*sample_crop_height)
        aspect_error = abs(grid_aspect-candidate_grid_aspect)
        if optimal_n_rows is None or aspect_error < optimal_aspect_error:
            optimal_n_rows = candidate_n_rows
            optimal_aspect_error = aspect_error
    
    assert optimal_n_rows is not None
    grid_rows = optimal_n_rows
    grid_columns = math.ceil(n_crops/grid_rows)        
    
    # Compute the width of each grid cell
    grid_cell_width = math.floor(grid_width / grid_columns)
    grid_cell_height = math.floor(grid_height / grid_rows)
    
    # Load secondary images and their associated bounding boxes. Iterate
    # through them, crop them, and save them to a list of cropped_images
    cropped_images = []
    for (name, box) in zip(secondary_image_filename_list,
                           secondary_image_bounding_box_list):
        
        other_image = vis_utils.load_image(name)
        cropped_image = crop_image_with_normalized_coordinates(
                other_image, box)
        
        # Rescale this crop to fit within the desired grid cell size
        width_scale_factor = grid_cell_width / cropped_image.size[0]
        height_scale_factor = grid_cell_height / cropped_image.size[1]
        scale_factor = min(width_scale_factor,height_scale_factor)
        
        # Resize the cropped image, whether we're making it larger or smaller
        cropped_image = cropped_image.resize(
                ((int)(cropped_image.size[0] * scale_factor),
                 (int)(cropped_image.size[1] * scale_factor)))            

        cropped_images.append(cropped_image)        

    # ...for each crop
    
    # Compute the final output image size. This will depend upon the aspect
    # ratio of the crops.
    output_image_width = primary_image.size[0] + grid_width
    output_image_height = primary_image.size[1]

    # Create blank output image
    output_image = Image.new('RGB', (output_image_width, output_image_height))

    # Copy resized primary image to output image
    if primary_image_location == 'right':
        primary_image_x = grid_width
    else:
        primary_image_x = 0
        
    output_image.paste(primary_image, (primary_image_x, 0))

    # Compute the final locations of the secondary images in the output image
    i_row = 0; i_col = 0
    for image in cropped_images:
        
        x = i_col * grid_cell_width
        if primary_image_location == 'left':
            x += primary_image.size[0]
        y = i_row * grid_cell_height 
        output_image.paste(image, (x,y))
        i_col += 1
        if i_col >= grid_columns:
            i_col = 0
            i_row += 1
            
    # ...for each crop

    # Write output image to disk
    output_image.save(output_image_filename)    


#%% Interactive driver

if False:
    
    pass

    #%%
    
    primary_image_filename = '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0035.JPG'
    
    primary_image_width = 5152
    
    secondary_image_filename_list = ['/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0035.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0040.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0007.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0041.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0008.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0048.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0031.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0006.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0004.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0026.JPG', '/home/user/data/KRU/KRU_public/KRU_S1/13/13_R1/KRU_S1_13_R1_IMAG0005.JPG']
    
    secondary_image_bounding_box_list = [[0, 0, 0.1853, 0.6552], [0, 0, 0.1855, 0.6527], [0, 0.000252, 0.1991, 0.6925], [0, 0, 0.1855, 0.6527], [0, 0.001008, 0.1902, 0.6774], [0, 0, 0.1845, 0.658], [0, 0, 0.1824, 0.6711], [0, 0.00252, 0.2005, 0.6857], [0, 0.002268, 0.1983, 0.6852], [0, 0, 0.1752, 0.6897], [0, 0.001764, 0.1989, 0.6887]]
    
    # cropped_grid_width = 3091
    cropped_grid_width = 500
    
    primary_image_location = 'right'

    output_image_filename = os.path.expanduser('~/tmp/grid-test.jpg')
    
    render_images_with_thumbnails(
            primary_image_filename,
            primary_image_width,
            secondary_image_filename_list,
            secondary_image_bounding_box_list,
            cropped_grid_width,
            output_image_filename,
            primary_image_location='right')    

    from md_utils import path_utils
    path_utils.open_file(output_image_filename)
    
    
#%% Command-line driver

def main():
    
    #%%
    
    from md_utils import path_utils
    
    # Load images from a test directory.
    #
    # Make the first image in the directory the primary image, 
    # the remaining ones the comparison images.    
    test_input_folder = os.path.expanduser('~/data/KRU-test')
    output_image_filename = os.path.expanduser('~/tmp/thumbnail_test.jpg')
    
    files = path_utils.find_images(test_input_folder)    

    random.seed(0); random.shuffle(files)
    primary_image_filename = files[0]

    secondary_image_filename_list = []
    secondary_image_bounding_box_list = []

    # Initialize the x,y location of the bounding box
    box = (random.uniform(0.25, 0.75), random.uniform(0.25, 0.75))

    # Create the list of secondary images and their bounding boxes
    for file in files[1:]:
        secondary_image_filename_list.append(file)
        secondary_image_bounding_box_list.append(
                (box[0] + random.uniform(-0.001, 0.001),
                 box[1] + random.uniform(-0.001, 0.001),
                 0.2,
                 0.2))
    
    primary_image_width = 1000
    cropped_grid_width = 1000

    render_images_with_thumbnails(
        primary_image_filename,
        primary_image_width,
        secondary_image_filename_list,
        secondary_image_bounding_box_list,
        cropped_grid_width,
        output_image_filename, 'right')
    
    from md_utils import path_utils
    path_utils.open_file(output_image_filename)
    
    #%%
    
if __name__ == '__main__':
    main()
