A demo to show converting caffe models on 'imagenet' using caffe2fluid

---

# How to use

1. Prepare python environment

2. Download caffe model to "models.caffe/xxx" which contains "xxx.caffemodel" and "xxx.prototxt"

3. Convert the Caffe model to Fluid model
    - generate fluid code and weight file
    <pre><code>python convert.py alexnet.prototxt \
        --caffemodel alexnet.caffemodel \
        --data-output-path alexnet.npy \
        --code-output-path alexnet.py
    </code></pre>

    - save weights as fluid model file
    <pre><code>python alexnet.py alexnet.npy ./fluid_model
    </code></pre>

4. Do inference
   <pre><code>python infer.py infer ./fluid_mode data/65.jpeg
</code></pre>

5. convert model and do inference together
   <pre><code>bash ./run.sh alexnet ./models.caffe/alexnet ./models/alexnet
</code></pre>
    The Caffe model is stored in './models.caffe/alexnet/alexnet.prototxt|caffemodel'
    and the Fluid model will be save in './models/alexnet/alexnet.py|npy'

6. test the difference with caffe's results(need pycaffe installed)
   <pre><code>bash ./diff.sh resnet
</code></pre>
Make sure your caffemodel stored in './models.caffe/resnet'.
The results will be stored in './results/resnet.paddle|caffe'
