import argparse
import distutils.util

def args_parser():
    parser = argparse.ArgumentParser(description='trainer file')
    parser.add_argument('--data_dir', type=str, default='flowers', help='source_directory')
    parser.add_argument('--dev',  type=str, default='gpu', help= 'gpu or cpu')
    parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='num of epochs')
    parser.add_argument('--arch', type=str, default='densenet121', help='other architectures are: densenet121, vgg16')
    parser.add_argument('--hidden_units1', type=int, default=512, help='hidden units for layer')
    parser.add_argument('--hidden_units2', type=int, default=256, help='hidden units for layer')
    parser.add_argument('--save_dir', type=str, default='densepoint.pth', help='save train model to a file')
     
    return parser.parse_args()