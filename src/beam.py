import time
import argparse
import math
import torch
import data
from vocabulary import Vocabulary
from model import V2S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path of the trained model file')
    parser.add_argument('--epoch', default=20)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--print_freq', default=50)
    parser.add_argument('--checkpoint_freq', default=2)
    parser.add_argument('--rnn_cell', default='lstm', type=str)
    parser.add_argument('--feature_size', default=4096, type=int)
    parser.add_argument('--embedding_size', default=500, type=int)
    parser.add_argument('--hidden_size', default=1000, type=int)
    parser.add_argument('--learning_rate', default=.0001, type=float, help='Initial learning rate.')
    parser.add_argument('--lr_update', default=5, type=int, help='Epoch frequency of decaying lr')
    parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
    parser.add_argument('--seq_maxlen', default=80)
    parser.add_argument('--num_val_decoding', default=2, help='Number of steps in which caption is printed')
    opt = parser.parse_args()

    data_path = '../data/'
    vocab_path = data_path+'msvd/coco_vocabulary.txt'
    vocab = Vocabulary(vocab_path)
    opt.vocab_size = len(vocab)

    testloader = data.get_dataloader(data_path, vocab, opt, 'test')

    # load pretrained model
    model = V2S(opt)
    model.load_state_dict(torch.load(opt.model_path))

if __name__ == '__main__':
    main()