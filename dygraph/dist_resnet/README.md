## Parallel Dygraph for the ResNet Training

- Launch the training job

    ``` bash
    > python -m paddle.distributed.launch --gpus 8 dist_train.py
    ```