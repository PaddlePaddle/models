wget http://www.vision.caltech.edu/visipedia-data/CUB-200/images.tgz
tar zxf images.tgz
find images|grep jpg|grep -v "\._" > list.txt
python split.py
rm -rf images.tgz list.txt
