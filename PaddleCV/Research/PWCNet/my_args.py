import argparse

parser = argparse.ArgumentParser(description='PWCNet_paddle')
parser.add_argument('--dataset', default='FlyingChairs', help='dataset type : FlyingChairs')
parser.add_argument('--data_root', default='', help='the path of selected datasets')
parser.add_argument('--model_out_dir', default='./out', help='the path of selected datasets')
parser.add_argument('--loss', default='l2', help='loss type : first train with l2 and finetune with l1')
parser.add_argument('--train_val_txt', default='', help='the path of selected train_val_txt of dataset')
parser.add_argument('--numEpoch', '-e', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--batch_size', '-b', type=int, default=40, help='batch size')
parser.add_argument('--pretrained', default=None, help='path to the pretrained model weights')
parser.add_argument('--optimize', default=None, help='path to the pretrained optimize weights')
parser.add_argument('--use_multi_gpu',action = 'store_true', help='Enable multi gpu mode')

args = parser.parse_args()
args.inference_size = [384, 512]
args.crop_size = [384, 448]