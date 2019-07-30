#!/bin/sh
# IWSLT15 Vietnames to English is a small dataset contain 133k parallel data
# this script download the data from stanford website
#
# Usage:
#   ./download_en-vi.sh output_path
#
# If output_path is not specified, a dir nameed "./en_vi" will be created and used as 
# output path

set -ex
OUTPUT_PATH="${1:-en-vi}"
SITE_PATH="https://nlp.stanford.edu/projects/nmt/data"

mkdir -v -p $OUTPUT_PATH

# Download iwslt15 small dataset from standford website.
echo "Begin to download training dataset train.en and train.vi."
wget "$SITE_PATH/iwslt15.en-vi/train.en" -O "$OUTPUT_PATH/train.en"
wget "$SITE_PATH/iwslt15.en-vi/train.vi" -O "$OUTPUT_PATH/train.vi"

echo "Begin to download dev dataset tst2012.en and tst2012.vi."
wget "$SITE_PATH/iwslt15.en-vi/tst2012.en" -O "$OUTPUT_PATH/tst2012.en"
wget "$SITE_PATH/iwslt15.en-vi/tst2012.vi" -O "$OUTPUT_PATH/tst2012.vi"

echo "Begin to download test dataset tst2013.en and tst2013.vi."
wget "$SITE_PATH/iwslt15.en-vi/tst2013.en" -O "$OUTPUT_PATH/tst2013.en"
wget "$SITE_PATH/iwslt15.en-vi/tst2013.vi" -O "$OUTPUT_PATH/tst2013.vi"

echo "Begin to ownload vocab file vocab.en and vocab.vi."
wget "$SITE_PATH/iwslt15.en-vi/vocab.en" -O "$OUTPUT_PATH/vocab.en"
wget "$SITE_PATH/iwslt15.en-vi/vocab.vi" -O "$OUTPUT_PATH/vocab.vi"

