import paddle.fluid.incubate.data_generator as dg
import numpy as np 
import paddle.fluid as fluid
from collections import defaultdict

all_field_id = ['101', '109_14', '110_14', '127_14', '150_14', '121', '122', '124', '125', '126', '127', '128', '129',
                '205', '206', '207', '210', '216', '508', '509', '702', '853', '301']
all_field_id_dict = defaultdict(int)
for i,field_id in enumerate(all_field_id):
    all_field_id_dict[field_id] = [False,i]

class CriteoDataset(dg.MultiSlotStringDataGenerator):
   
    def generate_sample(self, line):
        
        def reader():
            features = line.strip().split(',')
            #ctr = list(map(int, features[1]))
            #cvr = list(map(int, features[2]))
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
                #feat_id = list(map(int, feat_id))    
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

d = CriteoDataset()
d.run_from_stdin()