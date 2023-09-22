########
#
# gDrive_download.py
#
# Recursively enumerates a google drive directory, optionally downloading all the 
# files in that directory.  These are two separate steps; files are enumerated
# and written to file before anything is downloaded.  If you only want to connect
# file names and gDrive GUIDs, you don't need to download anything.
# 
# Uses the PyDrive library to talk to google drive, and assumes you've created a 
# .json file with your secret key to access the drive, following this tutorial 
# verbatim:
# 
# https://gsuitedevs.github.io/PyDrive/docs/build/html/quickstart.html#authentication
#
# It can take a few tries to run this on large data sets (in particular to
# retry failed downloads a semi-arbitrary number of times), so this isn't 
# entirely meant to be run from scratch; I'd say I ran this semi-interactvely.
#
# Note that gDrive caps free access at 1000 queries / 100 seconds / user = 
# 10 queries / second.  You may get slightly faster access than that in practice, but
# not much.
#
# agentmorris@gmail.com
#
########

#%% Imports

import time
import datetime
import json
import os
import csv
from pydrive.auth import GoogleAuth
from multiprocessing.pool import ThreadPool
from pydrive.drive import GoogleDrive
import humanfriendly


#%% Configuration and constants

# Should we actually download images, or just enumerate images?
downloadImages = 1

# Set to 'errors' when you've already downloaded most of the files and are just 
# re-trying failures
#
# 'all','errors','ifnecessary'
enumerationMode = 'ifnecessary'

# The GUID for the top-level folder
parentID = ''

# client_secrets.json lives here
clientSecretsPath = r'd:\git\danMisc'

# Limits the number of files we enumerate (for debugging).  Set to -1 to enumerate 
# all files.
maxFiles = -1

# This can be empty if we're not writing images
imageOutputDir = r'f:\video'

# When duplicate folders exist, should we merge them?  The alternative is
# renaming the second instance of "blah" to "blah (1)".  My experience has been
# that the gDrive sync behavior varies with OS; on Windows, renaming occurs, on MacOS,
# folders are merged.
bMergeDuplicateFolders = True


#%% Derived constants

# Change to the path where the client secrets file lives, to simplify auth
os.chdir(clientSecretsPath) 

# Create a datestamped filename to which we'll write all the metadata we
# retrieve when we crawl the gDrive.
metadataOutputDir = os.path.join(imageOutputDir,'metadata_cache')

os.makedirs(metadataOutputDir,exist_ok=True)

metadataFileBase = os.path.join(metadataOutputDir,'imageMetadata.json')
dateStamp = datetime.datetime.now().strftime('%Y.%m.%d.%H.%M.%S')
name, ext = os.path.splitext(metadataFileBase)
metadataFile = "{}.{}{}".format(name,dateStamp,ext)

# List of files we need to download, just filename and GUID.  This .csv
# file is written by the enumeration step.
downloadListFileBase = os.path.join(metadataOutputDir,'downloadList.csv')
name, ext = os.path.splitext(downloadListFileBase)
downloadListFile = "{}.{}{}".format(name,dateStamp,ext)

# List of download errors
errorListFileBase = os.path.join(metadataOutputDir,'enumerationErrors.csv')
name, ext = os.path.splitext(errorListFileBase)
errorListFile = "{}.{}{}".format(name,dateStamp,ext)

# If we are running in "errors" mode, this is the list of directories we want to re-try
errorListFileResume = os.path.join(metadataOutputDir,r"enumerationErrors.csv")

assert (not downloadImages) or (not len(imageOutputDir)==0), 'Can\'t have empty output dir if you\'re downloading images'

# Only applies to downloading; enumeration is not currently multi-threaded
nThreads = 10
    

#%% Authenticate

gauth = GoogleAuth()
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)


#%% Enumerate files for download (functions)

class DataEnumerator:

    nFiles = 0
    nFolders = 0
    errors = []
    fileInfo = []
    downloadList = []


def PrepareFolderDownload(folderID,folderTargetDir,dataEnumerator=None):
    """
    Enumerate files and directories in a single folder, specified by the GUID
    folderID.  Will be called once for every folder we encounter.  Does not make
    recursive calls.
    """
    
    if dataEnumerator == None:

        dataEnumerator = DataEnumerator()

    try:

        fileList = drive.ListFile({'q': "'%s' in parents and trashed=false" % folderID}).GetList()

    except Exception as ex:

        # ex = sys.exc_info()[0]
        errorString = str(ex)
        print("Error listing directory {}:{}:{}".format(folderTargetDir,folderID,errorString))
        dataEnumerator.errors.append( ['folder',folderTargetDir,folderID,errorString] )
        return dataEnumerator

    titles = set()

    # Handle redundant directory names
    for f in fileList:
        
        title = f['title']
        nRenames = 0
        
        if title in titles:            
            nRenames = nRenames + 1
            if bMergeDuplicateFolders:
                print("Warning: folder conflict at {}/{}".format(folderTargetDir,title))
            else:
                # Try to rename folders and files the way the gDrive sync app does, i.e. if there are 
                # two files called "Blah",  we want "Blah" and "Blah (1)".
                newTitle = title + " ({})".format(nRenames)
                print("Renaming {} to {} in [{}]".format(title,newTitle,folderTargetDir))
                title = newTitle
                f['title'] = title                
        else:
            titles.add(title)
            
    # ...for every file in our list (handling redundant directory names)
    
    # Enumerate and process files in this folder
    for f in fileList:
    
        if maxFiles > 0 and dataEnumerator.nFiles > maxFiles:
            return dataEnumerator

        dataEnumerator.fileInfo.append(f)

        title = f['title']
            
        if f['mimeType']=='application/vnd.google-apps.folder': # if folder

            dataEnumerator.nFolders = dataEnumerator.nFolders + 1
        
            # Create the target directory if necessary
            outputDir = os.path.join(folderTargetDir,title)
            f['target'] = outputDir
            if downloadImages:
                if not os.path.exists(outputDir):
                    os.mkdir(outputDir)

            print("Enumerating folder {} to {}".format(title,outputDir))        

            # Recurse
            dataEnumerator = PrepareFolderDownload(f['id'],outputDir,dataEnumerator)            
    
        else:            

            dataEnumerator.nFiles = dataEnumerator.nFiles + 1
        
            targetFile = os.path.join(folderTargetDir,title)            
            f['target'] = targetFile
            print("Downloading file {} to {}".format(title,targetFile))        
            dataEnumerator.downloadList.append( [targetFile,f['id']] )

    # ...for each file in this folder
    
    return dataEnumerator    

# ... def PrepareFolderDownload
    

#%% Enumerate files for download (execution)

startTime = time.time()

if (enumerationMode == 'ifnecessary') and (os.path.exists(downloadListFile)):

    downloadList = []
    with open(downloadListFile) as csvfile:
        r = csv.reader(csvfile)
        for iRow,row in enumerate(r):
            if maxFiles > 0 and iRow > maxFiles:
                break
            else:
                downloadList.append(row)

    print("Read {} downloads from {}".format(len(downloadList),downloadListFile))

else:

    dataEnumerator = None
        
    if enumerationMode == 'errors':

        splitLines = []

        assert(os.path.isfile(errorListFileResume))

        # Read the error file
        # For each line in the input file
        with open(errorListFileResume) as f:
            rows = csv.reader(f)    
            for iRow,row in enumerate(rows):
                splitLines.append(row)

        # Lines look like:
        # 
        # ['folder',folderTargetDir,folderID,errorString]

        for iRow,row in enumerate(splitLines):
            targetDir = row[1]
            folderID = row[2]
            errorString = row[3]
            print('Re-trying folder ID {} ({})'.format(targetDir,folderID))
            dataEnumerator = PrepareFolderDownload(folderID,targetDir)

    # Either we're in 'all' mode or we're in 'ifnecessary' mode and enumeration is necessary
    else:

        print("Starting enumeration")
        startTime = time.time()
        dataEnumerator = PrepareFolderDownload(parentID,imageOutputDir)
        elapsed = time.time() - startTime
        print("Finished enumeration in {}".format(str(datetime.timedelta(seconds=elapsed))))
    
        print("Enumerated {} files".format(len(dataEnumerator.downloadList)))

    s = json.dumps(dataEnumerator.fileInfo)
    with open(metadataFile, "w+") as f:
        f.write(s)
    print("Finished writing metadata to {}".format(metadataFile))

    with open(downloadListFile,'w+') as f:
        for fileInfo in dataEnumerator.downloadList:
            f.write(",".join(fileInfo) + "\n")
    print("Finished writing download list to {}".format(downloadListFile))

    with open(errorListFile,'w+') as f:
        for e in dataEnumerator.errors:
            f.write(",".join(e) + "\n")
    print("Finished writing error list ({} errors) to {}".format(len(dataEnumerator.errors),errorListFile))

    elapsed = time.time() - startTime
    print("Done enumerating files in {}".format(humanfriendly.format_timespan(elapsed)))

    downloadList = dataEnumerator.downloadList
    
# if/else on enumeration modes


#%% Compute total download size
    
import tqdm
import humanfriendly

sizeBytes = 0

for f in tqdm.tqdm(dataEnumerator.fileInfo):
    
    if 'fileSize' in f:
        sizeBytes = sizeBytes + int(f['fileSize'])
    
print('Total download size is {} in {} files'.format(
        humanfriendly.format_size(sizeBytes),len(dataEnumerator.fileInfo)))

    
#%% Download images (functions)

import sys

def ProcessDownload(fileInfo):

    status = 'unknown'
    targetFile = fileInfo[0]
    if os.path.exists(targetFile):
        print("Skipping download of file {}".format(targetFile))
        status = 'skipped'
        return status
    id = fileInfo[1]
    try:
        f = drive.CreateFile({'id': id})
        title = f['title']    
    except:
        print("File creation error for {}".format(targetFile))
        status = 'create_error'
        return status
    print("Downloading file {} to {}".format(title,targetFile))        
    try:
        f.GetContentFile(targetFile)
        status = 'success'
        return status
    except:
        print("Download error for {}: {}".format(targetFile,sys.exc_info()[0]))
        status = 'download_error'
        return status

def ProcessDownloadList(downloadList):

    pool = ThreadPool(nThreads)
    # results = pool.imap_unordered(lambda x: fetch_url(x,nImages), indexedUrlList)
    results = pool.map(ProcessDownload, downloadList)

    # for iFile,fileInfo in enumerate(downloadList):
    #    ProcessDownload(fileInfo)
    return results


#%% Download images (execution)

if downloadImages:

    print('Downloading data...')
    # results = ProcessDownloadList(downloadList[1:10])
    results = ProcessDownloadList(downloadList)
    print('...done.')
    

#%% Scrap

if False:

    #%% List files

    from pydrive.drive import GoogleDrive
    drive = GoogleDrive(gauth)
    file_list = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in file_list:
        print('title: %s, id: %s' % (file1['title'], file1['id']))


    #%% List a particular directory

    from pydrive.drive import GoogleDrive
    drive = GoogleDrive(gauth)
    folder_id = 'blahblahblah'
    q = {'q': "'{}' in parents and trashed=false".format(folder_id)}
    file_list = drive.ListFile(q).GetList()

    for iFile,f in enumerate(file_list):
        print('{}: {}, id: {}'.format(iFile,f['title'],f['id']))


    #%% Recursive list

    from pydrive.drive import GoogleDrive
    drive = GoogleDrive(gauth)

    def ListFolder(parentID,fileListOut=None):

        if  fileListOut is None:
            fileListOut = []

        parentList = drive.ListFile({'q': "'%s' in parents and trashed=false" % parentID}).GetList()

        for f in parentList:
    
            if len(fileListOut) > maxFiles:
                return fileListOut

            if f['mimeType']=='application/vnd.google-apps.folder': # if folder

                title = f['title']
                print("Enumerating folder {}".format(title))        
                childFiles = ListFolder(f['id'],fileListOut)
                print("Enumerated {} files".format(len(childFiles)))

                fileListOut = fileListOut + childFiles
                # fileListOut.append({"id":f['id'],"title":f['title'],"list":})
    
            else:            
                fileListOut.append(f['title'])

        return fileListOut

    parent = -1;
    file_list = ListFolder(parent)

    print("Enumerated {} files".format(len(file_list)))
