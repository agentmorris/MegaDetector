########
#
# create_links_to_md_results_files.py
#
# One-off script to populate the columns in the camera trap data .csv file that point to MD results.
#
########

#%% Imports and constants

import os

import pandas as pd

input_csv_file = r'g:\temp\lila_camera_trap_datasets_no_md_results.csv'
output_csv_file = r'g:\temp\lila_camera_trap_datasets.csv'

md_results_local_folder = r'g:\temp\lila-md-results'
md_base_url = 'https://lila.science/public/lila-md-results/'
assert md_base_url.endswith('/')

# No RDE files for datasets with no location information        
datasets_without_location_info = ('ena24','missouri-camera-traps')

md_results_column_names = ['mdv4_results_raw','mdv5a_results_raw','mdv5b_results_raw','md_results_with_rde']

validate_urls = False


#%% Read input data

df = pd.read_csv(input_csv_file)
for s in md_results_column_names:
    df[s] = ''
    
    
#%% Find matching files locally, and create URLs

local_files = os.listdir(md_results_local_folder)
local_files = [fn for fn in local_files if fn.endswith('.zip')]

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    
    if not isinstance(row['name'],str):
        continue
    
    dataset_shortname = row['short_name']
    matching_files = [fn for fn in local_files if dataset_shortname in fn]
    
    # No RDE files for datasets with no location information        
    if dataset_shortname in datasets_without_location_info:
        assert len(matching_files) == 2
        mdv5a_files = [fn for fn in matching_files if 'mdv5a' in fn]
        mdv5b_files = [fn for fn in matching_files if 'mdv5b' in fn]
        assert len(mdv5a_files) == 1 and len(mdv5b_files) == 1
        df.loc[i_row,'mdv5a_results_raw'] = md_base_url + mdv5a_files[0]
        df.loc[i_row,'mdv5b_results_raw'] = md_base_url + mdv5b_files[0]
    else:
        # Exclude single-season files for snpashot-serengeti    
        if dataset_shortname == 'snapshot-serengeti':
            matching_files = [fn for fn in matching_files if '_S' not in fn]
            assert len(matching_files) == 2        
            assert all(['mdv4' in fn for fn in matching_files])
            rde_files = [fn for fn in matching_files if 'rde' in fn]
            raw_files = [fn for fn in matching_files if 'rde' not in fn]
            assert len(rde_files) == 1 and len(raw_files) == 1
            df.loc[i_row,'mdv4_results_raw'] = md_base_url + raw_files[0]
            df.loc[i_row,'md_results_with_rde'] = md_base_url + rde_files[0]
        else:
            assert len(matching_files) == 3
            mdv5a_files = [fn for fn in matching_files if 'mdv5a' in fn and 'rde' not in fn]
            mdv5b_files = [fn for fn in matching_files if 'mdv5b' in fn and 'rde' not in fn]
            rde_files = [fn for fn in matching_files if 'rde' in fn]
            assert len(mdv5a_files) == 1 and len(mdv5b_files) == 1 and len(rde_files) == 1
            df.loc[i_row,'mdv5a_results_raw'] = md_base_url + mdv5a_files[0]
            df.loc[i_row,'mdv5b_results_raw'] = md_base_url + mdv5b_files[0]
            df.loc[i_row,'md_results_with_rde'] = md_base_url + rde_files[0]
            
    print('Found {} matching files for {}'.format(len(matching_files),dataset_shortname))

# ...for each row    


#%% Validate URLs

if validate_urls:
    
    from md_utils.url_utils import test_urls
    
    urls = set()
    
    for i_row,row in df.iterrows():
        for column_name in md_results_column_names:
            if len(row[column_name]) > 0:
                assert row[column_name] not in urls        
                urls.add(row[column_name])
                
    test_urls(urls,error_on_failure=True)            
    
    print('Validated {} URLs'.format(len(urls)))


#%% Write new .csv file

df.to_csv(output_csv_file,header=True,index=False)
