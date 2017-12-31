import time
import argparse
import math
import torch
import data
from vocabulary import Vocabulary
from model import V2S

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=20)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--print_freq', default=5)
parser.add_argument('--rnn_cell', default='lstm', type=str)
parser.add_argument('--feature_size', default=4096, type=int)
parser.add_argument('--embedding_size', default=500, type=int)
parser.add_argument('--hidden_size', default=1000, type=int)
parser.add_argument('--learning_rate', default=.0002, type=float, help='Initial learning rate.')
parser.add_argument('--grad_clip', default=2., type=float, help='Gradient clipping threshold.')
parser.add_argument('--seq_maxlen', default=80)
opt = parser.parse_args()

data_path = '../data/'
vocab_path = data_path+'msvd/coco_vocabulary.txt'
vocab = Vocabulary(vocab_path)
opt.vocab_size = len(vocab)

trainloader = data.get_dataloader(data_path, vocab, opt, 'debug')
valloader = data.get_dataloader(data_path, vocab, opt, 'debug')

# initialize the model
model = V2S(opt)
for i in range(opt.epoch):
    start = time.time()
    model.train()
    for j, (features, inputs, targets, videoids, lengths) in enumerate(trainloader):
        model.trainstep(features, inputs, targets, lengths)
        # prints training progression
        if (j+1) % opt.print_freq == 0:
            losses = model.loss_history[-opt.print_freq:-1]
            avglosses = float(sum(losses)) / len(losses)
            ppl = math.exp(avglosses)
            print 'Epoch: {:02d}, Iter: {:04d}/{:04d}, ' \
                  'Current loss: {:06.2f}, PPL: {:06.2f}, ' \
                  'Epoch time: {:06.2f}s'.format(i+1, j+1, len(trainloader),
                                                 avglosses, ppl, time.time()-start)

    # validating current model
    print '>>> Validating...'
    vallosses = []
    model.eval()
    for j, (features, inputs, targets, videoids, lengths) in enumerate(valloader):
        predsent, valloss = model.forward_eval(features, inputs, targets, lengths)
        vallosses.append(valloss)
    avgvallosses = float(sum(vallosses)) / len(vallosses)
    valppl = math.exp(avgvallosses)
    print '========================'
    print 'Epoch: {:02d}, Validation loss: {:06.2f},\tPPL: {:06.2f}'.format(i+1, avgvallosses, valppl)
    print '========================'
    print '>>> Decoding samples from the last batch...'
    for j, tup in enumerate(zip(vocab.translateSentences(targets, lengths),
                   vocab.translateSentences(predsent, lengths))):
        print '> tgt : {}\n> pred: {}'.format(tup[0], tup[1])
    print '========================'

    # training checkpointing