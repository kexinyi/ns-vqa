mkdir -p data/raw
cd data/raw

echo "Downloading CLEVR dataset..."
wget https://s3-us-west-1.amazonaws.com/clevr/CLEVR_v1.0.zip
unzip CLEVR_v1.0.zip
rm CLEVR_v1.0.zip

echo "Downloading CLEVR-mini dataset..."
wget http://nsvqa.csail.mit.edu/assets/CLEVR_mini.zip
unzip CLEVR_mini.zip
rm CLEVR_mini.zip

cd ..  # data
echo "Downloading pretrained model..."
wget http://nsvqa.csail.mit.edu/assets/pretrained.zip
unzip pretrained.zip
rm pretrained.zip

mkdir mask_rcnn
cd mask_rcnn
echo "Downloading pretrained weights for Mask R-CNN backbone..."
wget http://nsvqa.csail.mit.edu/assets/backbones.zip
unzip backbones.zip
rm backbones.zip

cd ..  # root