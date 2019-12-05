# compile cyops
python utils/cyops/setup.py develop

# compile and install pts_utils
cd utils/pts_utils
python setup.py install
cd ../..
