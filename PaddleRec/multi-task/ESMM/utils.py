import numpy as np 
import os
import paddle.fluid as fluid
import logging
from collections import defaultdict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

all_field_id = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125', '126', '127', '128', '129',
                '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
all_field_id_dict = defaultdict(int)
for i,field_id in enumerate(all_field_id):
    all_field_id_dict[field_id] = [False,i]

def get_dataset(inputs,files,batch_size,cpu_num):
    dataset = fluid.DatasetFactory().create_dataset()
    dataset.set_use_var(inputs)
    dataset.set_pipe_command("python dataset_generator.py")
    dataset.set_batch_size(batch_size)
    dataset.set_thread(int(cpu_num))
    file_list = [
        os.path.join(files, x) for x in os.listdir(files)
    ]
    logger.info("file list: {}".format(file_list))
    return dataset, file_list
    
def get_vocab_size(vocab_path):
    with open(vocab_path, "r") as rf:
        line = rf.readline()
        return int(line.strip()) + 1


class CriteoDataset(object):

    def _reader_creator(self, file):
        def reader():
            with open(file, 'r') as f:
                for line in f:
                    features = line.strip().split(',')
                    ctr = features[1]
                    cvr = features[2]
                    
                    padding = '0'
                    output = [(field_id,[]) for field_id in all_field_id_dict]
        
                    for elem in features[4:]:
                        field_id,feat_id = elem.strip().split(':')
                        if field_id not in all_field_id_dict:
                            continue
                        all_field_id_dict[field_id][0] = True
                        index = all_field_id_dict[field_id][1]
                        output[index][1].append(feat_id) 
                        
                    for field_id in all_field_id_dict:
                        visited,index = all_field_id_dict[field_id]
                        if visited:
                            all_field_id_dict[field_id][0] = False
                        else:
                            output[index][1].append(padding) 
                    output.append(('ctr',ctr))
                    output.append(('cvr',cvr))
                    yield output

        return reader

    def train(self, file):
        return self._reader_creator(file)

    def test(self, file):
        return self._reader_creator(file)




            