if [ -f assignment2.zip ]; then
    echo "data exist"
    exit 0
else
    wget http://cs224d.stanford.edu/assignment2/assignment2.zip
fi

if [ $? -eq 0  ];then
    unzip assignment2.zip
    cp assignment2_release/data/ner/wordVectors.txt .
    cp assignment2_release/data/ner/vocab.txt .
    rm -rf assignment2_release
else
  echo "download data error!" >> /dev/stderr
  exit 1
fi

