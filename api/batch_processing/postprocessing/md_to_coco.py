########
#
# md_to_coco.py
#
# "Converts" MegaDetector output files to COCO format.  "Converts" is in quotes because
# this is an opinionated transformation that requires a confidence threshold.
#
# Does not currently handle classification information.
#
########

#%% Constants and imports

import os
import json
import uuid

from tqdm import tqdm

from md_visualization import visualization_utils as vis_utils

default_confidence_threshold = 0.15


#%% Functions

def md_to_coco(md_results_file,
               coco_output_file=None,
               image_folder=None,
               confidence_threshold=default_confidence_threshold,
               validate_image_sizes=False,
               info=None,
               preserve_nonstandard_metadata=True,
               include_failed_images=True):
    """
    "Converts" MegaDetector output files to COCO format.  "Converts" is in quotes because
    this is an opinionated transformation that requires a confidence threshold.
    
    A folder of images is required if width and height information are not available 
    in the MD results file.
    
    If validate_image_sizes is True, we'll check the image sizes regardless of whether width
    and height are present in the MD results file.
    
    If preserve_nonstandard_metadata is True, confidence will be preserved in a non-standard 
    "conf" field in each annotation, and any random fields present in each image's data (e.g.
    EXIF metadata) will be propagated to COCO output.
    
    If include_failed_images is True, failed images will be propagated to COCO output with
    a non-empty "failure" field and no other fields, otherwise failed images will be skipped.
    
    Returns the COCO json dict.
    """

    with open(md_results_file,'r') as f:    
        md_results = json.load(f)    
        
    coco_images = []
    coco_annotations = []
    
    # im = md_results['images'][0]
    for im in tqdm(md_results['images']):
        
        coco_im = {}
        coco_im['id'] = im['file']
        coco_im['file_name'] = im['file']
        
        # There is no concept of this in the COCO standard        
        if 'failure' in im and im['failure'] is not None:
            if include_failed_images:
                coco_im['failure'] = im['failure']
                coco_images.append(coco_im)
            continue
        
        # Read/validate image size
        w = None
        h = None
        
        if ('width' not in im) or ('height' not in im) or validate_image_sizes:
            if image_folder is None:
                raise ValueError('Must provide an image folder when height/width need to be read from images')
            image_file_abs = os.path.join(image_folder,im['file'])
            pil_im = vis_utils.open_image(image_file_abs)
            w = pil_im.width
            h = pil_im.height
            if validate_image_sizes:
                if 'width' in im:
                    assert im['width'] == w, 'Width mismatch for image {}'.format(im['file'])
                if 'height' in im:
                    assert im['height'] == h, 'Height mismatch for image {}'.format(im['file'])
        else:
            w = im['width']
            h = im['height']
            
        coco_im['width'] = w
        coco_im['height'] = h
        
        # Add other, non-standard fields to the output dict
        if preserve_nonstandard_metadata:
            for k in im.keys():
                if k not in ('file','detections','width','height'):
                    coco_im[k] = im[k]
                
        coco_images.append(coco_im)
        
        # detection = im['detections'][0]
        for detection in im['detections']:
            
            # Skip below-threshold detections
            if confidence_threshold is not None and detection['conf'] < confidence_threshold:
                continue
        
            # Create an annotation
            ann = {}        
            ann['id'] = str(uuid.uuid1())
            ann['image_id'] = coco_im['id']                
                        
            md_category_id = detection['category']
            coco_category_id = int(md_category_id)
            ann['category_id'] = coco_category_id
            
            # In very esoteric cases, we use the empty category (0) in MD-formatted output files
            if md_category_id != '0':
                
                assert 'bbox' in detection,\
                    'Oops: non-empty category with no bbox in {}'.format(im['file'])
            
                ann['bbox'] = detection['bbox']
                # MegaDetector: [x,y,width,height] (normalized, origin upper-left)
                # COCO: [x,y,width,height] (absolute, origin upper-left)
                ann['bbox'][0] = ann['bbox'][0] * coco_im['width']
                ann['bbox'][1] = ann['bbox'][1] * coco_im['height']
                ann['bbox'][2] = ann['bbox'][2] * coco_im['width']
                ann['bbox'][3] = ann['bbox'][3] * coco_im['height']                
            
            else:
                
                print('Warning: empty category annotation in file {}'.format(im['file']))
                      
            if preserve_nonstandard_metadata:
                ann['conf'] = detection['conf']
                
            coco_annotations.append(ann)            
        
        # ...for each detection
    
    # ...for each image

    output_dict = {}
    
    if info is not None:
        output_dict['info'] = info
    else:
        output_dict['info'] = {'description':'Converted from MD results file {}'.format(md_results_file)}
    output_dict['info']['confidence_threshold'] = confidence_threshold
    
    output_dict['images'] = coco_images
    output_dict['annotations'] = coco_annotations
    
    output_dict['categories'] = []
    
    for md_category_id in md_results['detection_categories'].keys():
        
        coco_category_id = int(md_category_id)
        coco_category = {'id':coco_category_id,
                         'name':md_results['detection_categories'][md_category_id]}
        output_dict['categories'].append(coco_category)
        
    if coco_output_file is not None:
        with open(coco_output_file,'w') as f:
            json.dump(output_dict,f,indent=1)
    
    return output_dict

# def md_to_coco(...)


#%% Interactive driver

if False:
    
    pass

    #%% Configure options
    
    md_results_file = os.path.expanduser('~/data/md-test.json')
    coco_output_file = os.path.expanduser('~/data/md-test-coco.json')
    image_folder = os.path.expanduser('~/data/md-test')
    validate_image_sizes = True
    confidence_threshold = 0.2
    validate_image_sizes=True
    info=None
    preserve_nonstandard_metadata=True
    include_failed_images=False
    
    
    #%% Programmatic execution
    
    output_dict = md_to_coco(md_results_file,
                   coco_output_file=coco_output_file,
                   image_folder=image_folder,
                   confidence_threshold=confidence_threshold,
                   validate_image_sizes=validate_image_sizes,
                   info=info,
                   preserve_nonstandard_metadata=preserve_nonstandard_metadata,
                   include_failed_images=include_failed_images)
    
    
    #%% Command-line example
        
    s = f'python md_to_coco.py {md_results_file} {coco_output_file} {confidence_threshold} '
    if image_folder is not None:
        s += f' --image_folder {image_folder}'
    if preserve_nonstandard_metadata:
        s += ' --preserve_nonstandard_metadata'
    if include_failed_images:
        s += ' --include_failed_images'

    print(s); import clipboard; clipboard.copy(s)

    
    #%% Preview the resulting file
    
    from md_visualization import visualize_db
    options = visualize_db.DbVizOptions()
    options.parallelize_rendering = True
    options.viz_size = (900, -1)
    options.num_to_visualize = 5000

    html_file,_ = visualize_db.visualize_db(coco_output_file,
                                              os.path.expanduser('~/tmp/md_to_coco_preview'),
                                              image_folder,options)    

    from md_utils import path_utils # noqa
    path_utils.open_file(html_file)
    
    
#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='"Convert" MD output to COCO format, in quotes because this is an opinionated transformation that requires a confidence threshold')
    
    parser.add_argument(
        'md_results_file',
        type=str,
        help='Path to MD results file (.json)')
    
    parser.add_argument(
        'coco_output_file',
        type=str,
        help='Output filename (.json)')
    
    parser.add_argument(
        'confidence_threshold',
        type=float,
        default=default_confidence_threshold,
        help='Confidence threshold (default {})'.format(default_confidence_threshold)
        )
    
    parser.add_argument(
        '--image_folder',
        type=str,
        default=None,
        help='Image folder, only required if we will need to access image sizes'
        )
    
    parser.add_argument(
        '--preserve_nonstandard_metadata',
        action='store_true',
        help='Preserve metadata that isn\'t normally included in ' + 
             'COCO-formatted data (e.g. EXIF metadata, confidence values)'
        )

    parser.add_argument(
        '--include_failed_images',
        action='store_true',
        help='Keep a record of corrupted images in the output; may not be completely COCO-compliant'
        )
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    md_to_coco(args.md_results_file,
               args.coco_output_file,
               args.image_folder,
               args.confidence_threshold,
               validate_image_sizes=False,
               info=None,
               preserve_nonstandard_metadata=args.preserve_nonstandard_metadata,
               include_failed_images=args.include_failed_images)    
    
if __name__ == '__main__':
    main()
