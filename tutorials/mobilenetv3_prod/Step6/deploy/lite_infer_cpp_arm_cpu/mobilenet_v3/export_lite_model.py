from paddlelite.lite import *

def get_args(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(
        description='Paddle Lite Optimize model', add_help=add_help)

    parser.add_argument('--model-dir', default='mobilenet_v3_small', help='model dir')
    parser.add_argument('--model-file', default='', help='model file')
    parser.add_argument('--param-file', default='', help='param file')
    parser.add_argument('--target', default='arm', help='arm or opencl or X86')
    parser.add_argument('--model-type', default='naive_buffer', help='save model type')
    parser.add_argument('--optimize-out', default='mobilenet_v3_small', help='save model type')

    args = parser.parse_args()
    return args

def export(args):
    opt=Opt()
    opt.set_model_file(args.model_file)
    opt.set_param_file(args.param_file)
    opt.set_valid_places(args.target)
    opt.set_model_type(args.model_type)
    opt.set_optimize_out(args.optimize_out)
    opt.run()

if __name__ == "__main__":
    args = get_args()
    export(args)



