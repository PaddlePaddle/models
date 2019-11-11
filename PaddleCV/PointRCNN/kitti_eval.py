import os 
from tools.kitti_object_eval_python.evaluate import evaluate as kitti_evaluate 

def kitti_eval():
    label_dir = os.path.join('/home/ai/model/3d/train_data/KITTI/object/testing', 'label_2')
    split_file = os.path.join('/home/ai/model/3d/train_data/KITTI', 'ImageSets', 'val.txt')
    final_output_dir = os.path.join("./result_dir_pytorch", 'final_result', 'data')
    name_to_class = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
    ap_result_str, ap_dict = kitti_evaluate(
        label_dir, final_output_dir, label_split_file=split_file,
         current_class=name_to_class["Car"])
    print("KITTI evaluate: ", ap_result_str, ap_dict)


if __name__ == "__main__":
    kitti_eval()


