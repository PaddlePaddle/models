## Introduction
This is a Python module used to load and convert data into formats for detection model training, evaluation and inference. The converted sample schema is a tuple of np.ndarrays. For example, the schema of Faster R-CNN training data is: `[(im, im_info, im_id, gt_bbox, gt_class, is_crowd), (...)]`.

### Implementation
This module is consists of four sub-systems: data parsing, image pre-processing, data conversion and data feeding apis.  

We use `dataset.Dataset` to abstract a set of data samples. For example, `COCO` data contains 3 sets of data for training, validation, and testing respectively. Original data stored in files could be loaded into memory using `dataset.source`; Then make use of `dataset.transform` to process the data; Finally, the batch data could be fetched by the api of `dataset.Reader`.

Sub-systems introduction:
1. Data prasing     
By data parsing, we can get a `dataset.Dataset` instance, whose implementation is located in `dataset.source`. This sub-system is used to parse different data formats, which is easy to add new data format supports. Currently, only following data sources are included:

- COCO data source    
This kind of source is used to load `COCO` data directly, eg: `COCO2017`. It's composed of json files for labeling info and image files. And it's directory structure is as follows:

  ```
  data/coco/
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
  |   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  |   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  |   ...
  ```

- Pascal VOC data source       
This kind of source is used to load `VOC` data directly, eg: `VOC2007`. It's composed of xml files for labeling info and image files. And it's directory structure is as follows:


  ```
  data/pascalvoc/
  ├──Annotations
  │   ├── i000050.jpg
  │   ├── 003876.xml
  |   ...
  ├── ImageSets
  │   ├──Main
              └── train.txt
              └── val.txt
              └── test.txt
              └── dog_train.txt
              └── dog_trainval.txt
              └── dog_val.txt
              └── dog_test.txt
              └── ...
  │   ├──Layout
               └──...
  │   ├── Segmentation
                └──...
  ├── JPEGImages
  │   ├── 000050.jpg
  │   ├── 003876.jpg
  |   ...
  ```



- Roidb data source       
This kind of source is a normalized data format which only contains a pickle file. The pickle file only has a dictionary which only has a list named 'records' (maybe there is a mapping file for label name to label id named 'canme2id'). You can convert `COCO` or `VOC` data into this format.  The pickle file's content is as follows:
```python
(records, catname2clsid)
'records' is list of dict whose structure is:
{
    'im_file': im_fname, # image file name
    'im_id': im_id, # image id
    'h': im_h, # height of image
    'w': im_w, # width
    'is_crowd': is_crowd,
    'gt_class': gt_class,
    'gt_bbox': gt_bbox,
    'gt_poly': gt_poly,
}
'cname2id' is a dict to map category name to class id

```
We also provide the tool to generate the roidb data source in `./tools/`. You can use the follow command to implement.
```python 
# --type: the type of original data (xml or json)
# --annotation: the path of file, which contains the name of annotation files 
# --save-dir: the save path
# --samples: the number of samples (default is -1, which mean all datas in dataset)
python ./tools/generate_data_for_training.py 
            --type=json \
            --annotation=./annotations/instances_val2017.json \
            --save-dir=./roidb \
            --samples=-1 
```

 2. Image preprocessing     
 Image preprocessing subsystem includes operations such as image decoding, expanding, cropping, etc. We use `dataset.transform.operator` to unify the implementation, which is convenient for extension. In addition, multiple operators can be combined to form a complex processing pipeline, and used by data transformers in `dataset.transformer`, such as multi-threading to acclerate a complex image data processing.

 3. Data transformer     
 The function of the data transformer is used to convert a `dataset.Dataset` to a new `dataset.Dataset`, for example: convert a jpeg image dataset into a decoded and resized dataset. We use the decorator pattern to implement different transformers which are all subclass of `dataset.Dataset`. For example, the `dataset.transform.paralle_map` transformer is for multi-process preprocessing, more transformers can be found in `dataset.transform.transformer`.

 4. Data feeding apis     
To facilitate data pipeline building and data feeding for training, we combine multiple `dataset.Dataset` to form a `dataset.Reader` which can provide data for training, validation and testing respectively. The user only needs to call `Reader.[train|eval|infer]` to get the corresponding data stream. `Reader` supports yaml file to configure data address, preprocessing oprators, acceleration mode, and so on.



The main APIs are as follows:



1. Data parsing

 - `source/coco_loader.py`: Use to parse the COCO dataset. [detail code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/ppdet/data/source/coco_loader.py)
 - `source/voc_loader.py`: Use to parse the Pascal VOC dataset. [detail code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/ppdet/data/source/voc_loader.py)     
 [Note] When using VOC datasets, if you do not use the default label list, you need to generate `label_list.txt` using `tools/generate_data_for_training.py` (the usage method is same as generating the roidb data source) or provide `label_list.txt` in `data/pascalvoc/ImageSets/Main` firstly. Also set the parameter `use_default_label` to `false` in the configuration file.
 - `source/loader.py`: Use to parse the Roidb dataset. [detail code](https://github.com/PaddlePaddle/models/blob/develop/PaddleCV/object_detection/ppdet/data/source/loader.py)

2. Operator
 `transform/operators.py`: Contains a variety of data enhancement methods, including:

```  python
RandomFlipImage: Horizontal flip.
RandomDistort: Distort brightness, contrast, saturation, and hue.
ResizeImage: Adjust the image size according to the specific interpolation method.
RandomInterpImage: Use a random interpolation method to resize the image.
CropImage: Crop image with respect to different scale, aspect ratio, and overlap.
ExpandImage: Put the original image into a larger expanded image which is initialized using image mean.
DecodeImage: Read images in RGB format.
Permute: Arrange the channels of the image and converted to the BGR format.
NormalizeImage: Normalize image pixel values.
NormalizeBox: Normalize the bounding box.
MixupImage: Mixup two images in proportion.
```
[Note] The mixup operation can refer to[paper](https://arxiv.org/pdf/1710.09412.pdf)。

`transform/arrange_sample.py`: Sort the data which need to input the network.           
3. Transformer
`transform/post_map.py`: A pre-processing operation for completing batch data, which mainly includes:

```  python
Randomly adjust the image size of the batch data
Multi-scale adjustment of image size
Padding operation
```
`transform/transformer.py`: Used to filter useless data and return batch data.
`transform/parallel_map.py`: Used to achieve acceleration.          
4. Reader
`reader.py`: Used to combine source and transformer operations, and return batch data according to `max_iter`.
`data_feed.py`: Configure default parameters for `reader.py`.





### Usage

#### Ordinary usage
The function of this module is completed by combining the configuration information in the yaml file. The use of yaml files can be found in the configuration file section.

 - Read data for training

``` python
ccfg = load_cfg('./config.yml')
coco = Reader(ccfg.DATA, ccfg.TRANSFORM, maxiter=-1)
```
#### How to use customized dataset?
- Option 1: Convert the dataset to the VOC format or COCO format.
```python
 # In ./tools/, the code named labelme2coco.py is provided to convert
 # the dataset which is annotatedby Labelme to a COCO dataset.
 python ./tools/labelme2coco.py --json_input_dir ./labelme_annos/
                                --image_input_dir ./labelme_imgs/
                                --output_dir ./cocome/
                                --train_proportion 0.8
                                --val_proportion 0.2
                                --test_proportion 0.0
 # --json_input_dir：The path of json files which are annotated by Labelme.
 # --image_input_dir：The path of images.
 # --output_dir：The path of coverted COCO dataset.
 # --train_proportion：The train proportion of annatation data.
 # --val_proportion：The validation proportion of annatation data.
 # --test_proportion: The inference proportion of annatation data.
```
- Option 2:

1. Following the `./source/coco_loader.py` and `./source/voc_loader.py`, add `./source/XX_loader.py` and implement the `load` function.
2. Add the entry for `./source/XX_loader.py` in the `load` function of `./source/loader.py`.
3. Modify `./source/__init__.py`:


```python
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource']:
    source_type = 'RoiDbSource'
# Replace the above code with the following code:
if data_cf['type'] in ['VOCSource', 'COCOSource', 'RoiDbSource', 'XXSource']:
    source_type = 'RoiDbSource'
```

4. In the configure file, define the `type` of `dataset` as `XXSource`。  

#### How to add data pre-processing？
- If you want to add the enhanced preprocessing of a single image, you can refer to the code of each class in `transform/operators.py`, and create a new class to implement new data enhancement. Also add the name of this preprocessing to the configuration file.
- If you want to add image preprocessing for a single batch, you can refer to the code for each function in `build_post_map` of `transform/post_map.py`, and create a new internal function to implement new batch data preprocessing. Also add the name of this preprocessing to the configuration file.
