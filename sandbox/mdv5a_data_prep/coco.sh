mkdir coco
cd coco
mkdir images
cd images

wget -c http://images.cocodataset.org/zips/train2017.zip

unzip train2017.zip

rm train2017.zip

cd ../

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip annotations_trainval2017.zip

rm annotations_trainval2017.zip

