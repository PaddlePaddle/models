import paddle
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

model_link = {
    'vit_h':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_h/model.pdparams",
    'vit_l':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_l/model.pdparams",
    'vit_b':
    "https://bj.bcebos.com/paddleseg/dygraph/paddlesegAnything/vit_b/model.pdparams"
}


def build_predictor():
    print("Loading model...")

    if paddle.is_compiled_with_cuda():
        paddle.set_device("gpu")
    else:
        paddle.set_device("cpu")

    sam = sam_model_registry["vit_b"](checkpoint=model_link["vit_b"])
    generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask")

    return generator
