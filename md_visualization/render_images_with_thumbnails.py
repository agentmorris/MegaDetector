########
# 
# render_images_with_thumbnails.py
# 
# Render an output image with one primary and many secondary images,
# used to check if suspicious detections are actually false positives or not.
#
########

#%% Constants and imports

import math
import os
import random

from md_visualization import visualization_utils as vis_utils
from PIL import Image

#%% Functions
def crop_image_with_normalized_coordinates(
        image,
        bounding_box):
    """
    Args:
        image: image to crop
        bounding_box: tuple formatted as (x,y,w,h), where (0,0) is the
        upper-left of the image, and coordinates are normalized
        (so (0,0,1,1) is a box containing the entire image.
    """
    im_width, im_height = image.size
    (x_norm, y_norm, w_norm, h_norm) = bounding_box
    (x, y, w, h) = (x_norm * im_width,
                    y_norm * im_height,
                    w_norm * im_width,
                    h_norm * im_height)
    return image.crop((x, y, x+w, y+h))


def render_images_with_thumbs(
        primary_image_filename,
        primary_image_width,
        secondary_image_filename_list,
        secondary_image_bounding_box_list,
        cropped_grid_width,
        output_image_filename):
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
        primary_image_width: width at which to render the primary image,
        secondary_image_filename_list: list of strs that are the filenames of
            the secondary images.
        secondary_image_bounding_box_list: list of tuples, one per secondary
            image. Each tuple is a bounding box of the secondary image,
            formatted as (x,y,w,h), where (0,0) is the upper-left of the image,
            and coordinates are normalized (so (0,0,1,1) is a box containing
            the entire image.
        cropped_grid_width: width of all the cropped images
        output_image_filename: str of the filename to write the output image
    """

    # Check to make sure the arguments are reasonable
    assert(len(secondary_image_filename_list) ==
           len(secondary_image_bounding_box_list))

    # Compute the number of grid elements for the secondary images
    # To make things easy, turn the secondary images into a 
    # n x n grid
    grid_count = math.ceil(math.sqrt(len(secondary_image_filename_list)))
    print(f'Grid count is {grid_count}')

    # Compute the width of each grid. 
    grid_width = math.floor(cropped_grid_width / grid_count)
    print(f'Grid width is {grid_width}')

    # Load primary image and resize to desired width
    primary_image = vis_utils.load_image(primary_image_filename)
    print(primary_image.size)
    primary_image = vis_utils.resize_image(
            primary_image, primary_image_width, -1)
    print(primary_image.size)

    # Load secondary images and their associated bounding boxes. Iterate
    # through them, crop them, and save them to a list of cropped_images
    cropped_images = []
    max_cropped_image_height = 0
    max_cropped_image_width = 0
    for (name, box) in zip(secondary_image_filename_list,
                           secondary_image_bounding_box_list):
        print(f'{name} has {box}')
        other_image = vis_utils.load_image(name)
        cropped_image = crop_image_with_normalized_coordinates(
                other_image, box)
        print(f'Original cropped size {cropped_image.size}')
        print(f'Aspect ratio is {cropped_image.size[0]/cropped_image.size[1]}')

        # Rescale the images to fit within the desired grid_width if the
        # crop is too big.
        # Note we probably could have used vis_utils.resize_image() instead
        # of doing this ourselves.
        scale_factor = cropped_image.size[0] / grid_width
        print(f'scale factor is {scale_factor}')
        if scale_factor >= 1: # only resize if image is too big
            cropped_image = cropped_image.resize(
                    ((int)(cropped_image.size[0] / scale_factor),
                     (int)(cropped_image.size[1] / scale_factor)))
            print(f'Rescaled crop to {cropped_image.size}')

        cropped_images.append(cropped_image)

        # Record the maximum width/height of the cropped images for later
        if cropped_image.size[0] > max_cropped_image_width:
            max_cropped_image_width = cropped_image.size[0]
        if cropped_image.size[1] > max_cropped_image_height:
            max_cropped_image_height = cropped_image.size[1]


    # Compute the final output image size. This will depend upon the aspect
    # ratio of the crops.
    output_image_width = primary_image.size[0] + cropped_grid_width
    output_image_height = max(
            primary_image.size[1], max_cropped_image_height*grid_count)
    print(f'output_image is {output_image_width} x {output_image_height}')

    # Create blank output image.
    output_image = Image.new("RGB", (output_image_width, output_image_height))

    # Copy resized primary image to output image
    output_image.paste(primary_image, (max_cropped_image_width*grid_count, 0))

    # Compute the final locations of the secondary images in the output image
    m = n = 0 # initialize grid coordinates to zero
    for image in cropped_images:
        x = m * grid_width 
        y = n * max_cropped_image_height 
        print(f'{m},{n} position is {x,y}')
        output_image.paste(image, (x,y))
        m += 1
        if m >= grid_count:
            m = 0
            n += 1

    # Write output image to disk
    output_image.show()
    output_image.save(output_image_filename)


def square_crops(filename, num_secondary):
    """
    Helper function for testing. 

    Given a filename and number, returns a tuple of two lists of the
    same length. The first element of the tuple is a list of the
    filename repeated, the second element of the tuple is
    a list of bounding boxes. 

    In this case the bounding boxes are squares of the same size,
    going down the diagonal of the image (from top left to bottom right)
    """
    secondary_image_filenames = []
    secondary_image_bounding_boxes = []
    for x in range(num_secondary):
        secondary_image_filenames.append(filename)
        box = (x/num_secondary, x/num_secondary,
               1/num_secondary, 1/num_secondary)
        print(box)
        secondary_image_bounding_boxes.append(box)
   
    return (secondary_image_filenames,
            secondary_image_bounding_boxes)


def wide_crops(filename, num_secondary):
    """
    Helper function for testing. 

    Given a filename and number, returns a tuple of two lists of the
    same length. The first element of the tuple is a list of the
    filename repeated, the second element of the tuple is
    a list of bounding boxes. 

    Actually returns a list of num_secondary-1 since we're asking for wider
    images.
    """
    secondary_image_filename_list = []
    secondary_image_bounding_box_list = []
    for x in range(1, num_secondary):
        secondary_image_filename_list.append(filename)
        box = (x/num_secondary-(1/num_secondary), x/num_secondary,
               2/num_secondary, 1/num_secondary)
        print(box)
        secondary_image_bounding_box_list.append(box)
    return (secondary_image_filename_list,
            secondary_image_bounding_box_list)


def tall_crops(filename, num_secondary):
    """
    Helper function for testing. 

    Given a filename and number, returns a tuple of two lists of the
    same length. The first element of the tuple is a list of the
    filename repeated, the second element of the tuple is
    a list of bounding boxes. 

    Actually returns a list of num_secondary-1 since we're asking for taller
    images.

    """
    secondary_image_filename_list = []
    secondary_image_bounding_box_list = []
    for x in range(1, num_secondary):
        secondary_image_filename_list.append(filename)
        box = (x/num_secondary, x/num_secondary-(1/num_secondary),
               1/num_secondary, 2/num_secondary)
        print(box)
        secondary_image_bounding_box_list.append(box)
    return (secondary_image_filename_list,
            secondary_image_bounding_box_list)


def main():
    # Load images from test directory (which we know has 31 images).
    # Make the first image in the directory the primary image, 
    # the remaining ones the comparison images.
    files = os.listdir("MTZ")

    # Filter out non JPG files
    files = [x for x in files if x.endswith("JPG")]

    random.seed(3)
    random.shuffle(files)

    primary_image_filename = "MTZ/" + files[0]

    # don't save the output file in MTZ otherwise from run-to-run
    # we keep getting more images in our test directory
    output_image_filename = "output_" + files[0]
    
    secondary_image_filename_list = []
    secondary_image_bounding_box_list = []

    # initialize the x,y location of the bounding box
    box = (random.uniform(0.25, 0.75), random.uniform(0.25, 0.75))

    # create the list of secondary images and their bounding boxes
    for file in files[1:]:
        secondary_image_filename_list.append("MTZ/"+file)
        secondary_image_bounding_box_list.append(
                (box[0] + random.uniform(-0.001, 0.001),
                 box[1] + random.uniform(-0.001, 0.001),
                 0.2,
                 0.2))
    
    primary_image_width = 1000
    cropped_grid_width = 1000

    render_images_with_thumbs(
        primary_image_filename,
        primary_image_width,
        secondary_image_filename_list,
        secondary_image_bounding_box_list,
        cropped_grid_width,
        output_image_filename)


if __name__ == '__main__':
    main()
