"""

 zamba_results_to_md_results.py

 Convert a labels.csv file produced by Zamba Cloud to a MD results file suitable
 for import into Timelapse.

 Columns are expected to be:

 video_uuid (not used)
 original_filename (assumed to be a relative path name)
 top_k_label,top_k_probability, for k = 1..N
 [category name 1],[category name 2],...
 corrected_label

 Because the MD results file fundamentally stores detections, what we'll
 actually do is created bogus detections that fill the entire image.  Detection
 coordinates are not currently used in Timelapse video video anyway.

 There is no special handling of empty/blank categories; because these results are
 based on a classifier, rather than a detector (where "blank" would be the absence of
 all other categories), "blank" can be queried in Timelapse just like any other class.

"""

#%% Imports and constants

import pandas as pd
import json


#%% Main function

def zamba_results_to_md_results(input_file,output_file=None):
    """
    Converts the .csv file [input_file] to the MD-formatted .json file [output_file].
    
    If [output_file] is None, '.json' will be appended to the input file.
    """
    
    if output_file is None:
        output_file = input_file + '.json'

    df = pd.read_csv(input_file)
    
    expected_columns = ('video_uuid','corrected_label','original_filename')
    for s in expected_columns:
        assert s in df.columns,\
            'Expected column {} not found, are you sure this is a Zamba results .csv file?'.format(
                s)
            
    # How many results are included per file?
    assert 'top_1_probability' in df.columns and 'top_1_label' in df.columns
    top_k = 2
    while(True):
        p_string = 'top_' + str(top_k) + '_probability'
        label_string = 'top_' + str(top_k) + '_label'
        
        if p_string in df.columns:
            assert label_string in df.columns,\
                'Oops, {} is a column but {} is not'.format(
                    p_string,label_string)
            top_k += 1
            continue
        else:
            assert label_string not in df.columns,\
                'Oops, {} is a column but {} is not'.format(
                    label_string,p_string)
            top_k -= 1
            break
            
    print('Found {} probability column pairs'.format(top_k))
    
    # Category names start after the fixed columns and the probability columns
    category_names = []
    column_names = list(df.columns)
    first_category_name_index = 0
    while('top_' in column_names[first_category_name_index] or \
          column_names[first_category_name_index] in expected_columns):
        first_category_name_index += 1
        
    i_column = first_category_name_index
    while( (i_column < len(column_names)) and (column_names[i_column] != 'corrected_label') ):
        category_names.append(column_names[i_column])
        i_column += 1
        
    print('Found {} categories:\n'.format(len(category_names)))
    
    for s in category_names:
        print(s)
    
    info = {}
    info['format_version'] = '1.3'
    info['detector'] = 'Zamba Cloud'
    info['classifier'] = 'Zamba Cloud'    
    
    detection_category_id_to_name = {}
    for category_id,category_name in enumerate(category_names):
        detection_category_id_to_name[str(category_id)] = category_name
    detection_category_name_to_id = {v: k for k, v in detection_category_id_to_name.items()}
    
    images = []
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
        
        im = {}
        images.append(im)
        im['file'] = row['original_filename']
        
        detections = []
        
        # k = 1
        for k in range(1,top_k+1):
            label = row['top_{}_label'.format(k)]
            confidence = row['top_{}_probability'.format(k)]
            det = {}
            det['category'] = detection_category_name_to_id[label]
            det['conf'] = confidence
            det['bbox'] = [0,0,1.0,1.0]
            detections.append(det)
            
        im['detections'] = detections
        
    # ...for each row
    
    results = {}
    results['info'] = info
    results['detection_categories'] = detection_category_id_to_name
    results['images'] = images
    
    with open(output_file,'w') as f:
        json.dump(results,f,indent=1)
        
# ...zamba_results_to_md_results(...)
        

#%% Interactive driver

if False:

    pass

    #%%
    
    input_file = r"G:\temp\labels-job-b95a4b76-e332-4e17-ab40-03469392d36a-2023-11-04_16-28-50.060130.csv"
    output_file = None
    zamba_results_to_md_results(input_file,output_file)
    

#%% Command-line driver

import sys,argparse

def main():

    parser = argparse.ArgumentParser(
        description='Convert a Zamba-formatted .csv results file to a MD-formatted .json results file')
    
    parser.add_argument(
        'input_file',
        type=str,
        help='input .csv file')
    
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='output .json file (defaults to input file appended with ".json")')
    
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    zamba_results_to_md_results(args.input_file,args.output_file)
    
if __name__ == '__main__':
    main()
    
