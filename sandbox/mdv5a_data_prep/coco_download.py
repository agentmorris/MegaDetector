import os
import urllib.request
import zipfile

os.makedirs('coco/images', exist_ok=True)

image_urls = [
    'http://images.cocodataset.org/zips/train2017.zip',
    'http://images.cocodataset.org/zips/val2017.zip'
    'http://images.cocodataset.org/zips/test2017.zip'
]

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    print(f"\rDownload progress: {percent}%", end='')

for url in image_urls:
    filename = url.split('/')[-1]
    filepath = os.path.join('coco/images', filename)
    urllib.request.urlretrieve(url, filepath, download_progress)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(filepath))
    os.remove(filepath)

annotation_urls = [
    'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
]

for url in annotation_urls:
    filename = url.split('/')[-1]
    filepath = os.path.join('coco', filename)
    urllib.request.urlretrieve(url, filepath, download_progress)

    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall('coco')
    os.remove(filepath)
