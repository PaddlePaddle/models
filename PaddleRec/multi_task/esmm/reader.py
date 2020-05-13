import numpy as np
import pandas as pd
from collections import defaultdict
import args
import os

def join_data(file1,file2,write_file,sample_size):
    sample_list = []
    common_logs = defaultdict(lambda: '')
    file = open(write_file, 'w')  
    
    print("begin push sample_list!")
    with open(file1,'r') as f:
        for i, line in enumerate(f):
            try:
                sample_list.append(line)
            except:
                continue
            
    print("begin push common_logs!")        
    with open(file2,'r') as f:
        for i, line in enumerate(f):
            try:
                common_feature_index,sample_str = line.strip().split('\t')
                common_logs[common_feature_index] = sample_str
            except:
                continue
            
    print("begin join data!") 
    for i, sample in enumerate(sample_list):
        try:
            common_feature_index,sample_str = sample.strip().split('\t')
            common_str = common_logs.get(common_feature_index)
            if common_str:
                sample =  "{0},{1}".format(sample_str, common_str)
            else:
                sample = "{0}".format(sample_str)
            file.write(sample + "\n")
        except:
            continue
        if(i == sample_size):
            break
    
    print("join data successfully!")


def read_data(file_name,write_file):
    file = open(write_file, 'w')  
    print("begin to write!")
    with open(file_name,'r') as f:
        for i, line in enumerate(f):
            try:
                line = line.strip().split(',')
                feat_len = len(line)
                feat_lists = []
                #common_feature_index|feat_num|feat_list
                if(feat_len == 3):
                    feat_strs = line[2]
                    for fstr in feat_strs.split('\x01'):
                        filed, feat_val = fstr.split('\x02')
                        feat, val = feat_val.split('\x03')
                        feat_lists.append('%s:%s' % (filed,feat))
                    common_feature = "{0}\t{1}".format(line[0], ','.join(feat_lists)) + "\n"
                    file.write(common_feature)
                
                #sample_id|y|z|common_feature_index|feat_num|feat_list
                elif(feat_len == 6):
                    # y=0 & z=1 filter
                    if(line[1] == '0' and line[2] == '1'):
                        continue
                    feat_strs = line[5]
                    for fstr in feat_strs.split('\x01'):
                        filed, feat_val = fstr.split('\x02')
                        feat, val = feat_val.split('\x03')
                        feat_lists.append('%s:%s' % (filed,feat))
                    sample = "{0}\t{1},{2},{3},{4}".format(line[3], line[0], line[1], line[2], ','.join(feat_lists)) + "\n"
                    file.write(sample)
            except:
                continue
           
    file.close()

def recode(file_path,writh_file,vocab_path):
    all_feat_id_dict = defaultdict(int)
    file1 = open(writh_file[0], 'w') 
    file2 = open(writh_file[1], 'w')
    vocab_file = open(vocab_path, 'w') 
    id = 0
    with open(file_path[0], "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(',')
            feat_lists = []
            for elem in line[3:]:
                field_id,feat_id = elem.strip().split(':')
                if feat_id not in all_feat_id_dict:
                    id += 1
                    all_feat_id_dict[feat_id] = id
                feat_lists.append('%s:%s' % (field_id,all_feat_id_dict[feat_id]))
            sample = "{0},{1},{2},{3}".format(line[0], line[1], line[2], ','.join(feat_lists)) + "\n"
            file1.write(sample)
    with open(file_path[1], "r") as f:
        for i, line in enumerate(f):
            line = line.strip().split(',')
            feat_lists = []
            for elem in line[3:]:
                field_id,feat_id = elem.strip().split(':')
                if feat_id not in all_feat_id_dict:
                    id += 1
                    all_feat_id_dict[feat_id] = id
                feat_lists.append('%s:%s' % (field_id,all_feat_id_dict[feat_id]))
            sample = "{0},{1},{2},{3}".format(line[0], line[1], line[2], ','.join(feat_lists)) + "\n"
            file2.write(sample)
    vocab_size =len(all_feat_id_dict) 
    vocab_file.write(str(vocab_size))
    file1.close()
    file2.close()
    vocab_file.close()
            
if __name__ == "__main__":
    args = args.parse_args()
    
    read_data(args.train_data_path + '/sample_skeleton_train.csv',args.train_data_path + '/skeleton_train.csv')
    print("write skeleton_train.csv successfully")
    read_data(args.train_data_path + '/common_features_train.csv',args.train_data_path + '/features_train.csv')
    print("write features_train.csv successfully")
    
    skeleton_train_path = args.train_data_path + '/skeleton_train.csv'
    features_train_path = args.train_data_path + '/features_train.csv'
    
    write_file = args.train_data_path + '/train_data.csv'
    join_data(skeleton_train_path,features_train_path,write_file,args.train_sample_size)

    os.system('rm -rf ' + skeleton_train_path)
    os.system('rm -rf ' + features_train_path)
    
    
    read_data(args.test_data_path + '/sample_skeleton_test.csv',args.test_data_path + '/skeleton_test.csv')
    print("write skeleton_est.csv successfully")
    read_data(args.test_data_path + '/common_features_test.csv',args.test_data_path + '/features_test.csv')
    print("write features_test.csv successfully")
    
    skeleton_test_path = args.test_data_path + '/skeleton_test.csv'
    features_test_path = args.test_data_path + '/features_test.csv'
    
    write_file = args.test_data_path + '/test_data.csv'
    join_data(skeleton_test_path,features_test_path,write_file,args.test_sample_size)

    os.system('rm -rf ' + skeleton_test_path)
    os.system('rm -rf ' + features_test_path)
    
    
    file_path = [args.train_data_path + '/train_data.csv', args.test_data_path + '/test_data.csv']
    write_file = [args.train_data_path + '/traindata.csv',args.test_data_path + '/testdata.csv']
    recode(file_path,write_file,args.vocab_path)

    for file in file_path:
        os.system('rm -rf ' + file_path)
