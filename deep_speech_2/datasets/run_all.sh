export PYTHONPATH=`pwd`:$PYTHONPATH
cd datasets

python thchs30/thchs30.py
if [ $? -ne 0 ]; then
    echo "Prepare LHCHS30 failed. Terminated."
    exit 1
fi

python librispeech/librispeech.py
if [ $? -ne 0 ]; then
    echo "Prepare LibriSpeech failed. Terminated."
    exit 1
fi

cat librispeech/manifest.train* | shuf > manifest.train
cat librispeech/manifest.dev-clean > manifest.dev
cat librispeech/manifest.test-clean > manifest.test

echo "All done."

cd -
