########
#
# process_utils.py
#
# Run something at the command line and capture the output, based on:
#
# https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
# 
# Includes handy example code for doing this on multiple processes/threads.
#
########

#%% Constants, imports, and environment

import os
import subprocess

os.environ["PYTHONUNBUFFERED"] = "1"

def execute(cmd):
    """
    Run [cmd] (a single string) in a shell, yielding each line of output to the caller.
    """
 
    # https://stackoverflow.com/questions/4417546/constantly-print-subprocess-output-while-process-is-running
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             shell=True, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute_and_print(cmd,print_output=True):
    """
    Run [cmd] (a single string) in a shell, capturing and printing output.  Returns
    a dictionary with fields "status" and "output".
    """

    to_return = {'status':'unknown','output':''}
    output=[]
    try:
        for s in execute(cmd):
            output.append(s)
            if print_output:
                print(s,end='',flush=True)
        to_return['status'] = 0
    except subprocess.CalledProcessError as cpe:
        print('execute_and_print caught error: {}'.format(cpe.output))
        to_return['status'] = cpe.returncode
    to_return['output'] = output
   
    return to_return


#%% Single-threaded test driver for execute_and_print

if False:
    
    pass

    #%%
    
    if os.name == 'nt':
        execute_and_print('echo hello && ping -n 5 127.0.0.1 && echo goodbye')  
    else:
        execute_and_print('echo hello && sleep 1 && echo goodbye')  
 

#%% Parallel test driver for execute_and_print

if False:
   
    pass

    #%%
   
    from functools import partial
    from multiprocessing.pool import ThreadPool as ThreadPool
    from multiprocessing.pool import Pool as Pool
   
    n_workers = 10
   
    # Should we use threads (vs. processes) for parallelization?
    use_threads = True
   
    test_data = ['a','b','c','d']
   
    def process_sample(s):
        return execute_and_print('echo ' + s,True)
       
    if n_workers == 1:  
     
        results = []
        for i_sample,sample in enumerate(test_data):    
            results.append(process_sample(sample))
     
    else:
     
        n_threads = min(n_workers,len(test_data))
     
        if use_threads:
            print('Starting parallel thread pool with {} workers'.format(n_threads))
            pool = ThreadPool(n_threads)
        else:
            print('Starting parallel process pool with {} workers'.format(n_threads))
            pool = Pool(n_threads)
   
        results = list(pool.map(partial(process_sample),test_data))
      
        for r in results:
            print(r)
