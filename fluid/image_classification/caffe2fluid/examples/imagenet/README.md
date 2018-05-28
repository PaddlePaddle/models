A demo to show converting caffe models trained on 'imagenet' using caffe2fluid

---

# How to use

1. Prepare python environment

2. Download caffe model to "models.caffe/xxx" which contains "xxx.caffemodel" and "xxx.prototxt"

3. Convert the Caffe model to Fluid model
    - generate fluid code and weight file
        ```python convert.py alexnet.prototxt \
        --caffemodel alexnet.caffemodel \
        --data-output-path alexnet.npy \
        --code-output-path alexnet.py
        ```

    - save weights as fluid model file
        ```
        python alexnet.py alexnet.npy ./fluid
        ```

4. Do inference
    ```
    python infer.py infer ./fluid data/65.jpeg
    ```

5. convert model and do inference together
   ```
    bash ./tools/run.sh alexnet ./models.caffe/alexnet ./models/alexnet
    ```
    * Assume the Caffe model is stored in '*./models.caffe/alexnet/alexnet.prototxt|caffemodel*'
    * converted model will be stored as '*./models/alexnet/alexnet.py|npy*'

6. test the difference with caffe's results(need pycaffe installed)
   ```
    bash ./tools/diff.sh resnet
    ```
    * Make sure your caffemodel stored in '*./models.caffe/resnet*'
    * The results will be stored in '*./results/resnet.paddle|caffe*'
