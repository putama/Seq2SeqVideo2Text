import time
import argparse
import math
import torch
import data
from vocabulary import Vocabulary
from model import V2S

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=20)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--print_freq', default=50)
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

    trainloader = data.get_dataloader(data_path, vocab, opt, 'train')
    valloader = data.get_dataloader(data_path, vocab, opt, 'val')

    # initialize the model
    model = V2S(opt)
    for i in range(opt.epoch):
        # decay learning rate
        adjust_learning_rate(opt, model.optimizer, i)
        start = time.time()
        model.train()
        for j, (features, inputs, targets, videoids, lengths) in enumerate(trainloader):
            model(features, inputs, targets, lengths, phase='train')
            # prints training progression
            if (j+1) % opt.print_freq == 0:
                losses = model.loss_history[-opt.print_freq:-1]
                avglosses = float(sum(losses)) / len(losses)
                ppl = math.exp(avglosses)
                print 'Epoch: {:02d}, Iter: {:04d}/{:04d}, ' \
                      'Current loss: {:06.2f}, PPL: {:06.2f}, ' \
                      'Epoch time: {:06.2f}s, Lr: {:08.7f}'.format(model.epoch_count+1, j+1,
                            len(trainloader), avglosses, ppl,
                            time.time()-start, opt.learning_rate)
        model.epoch_count = model.epoch_count + 1

        # validating current model
        print '>>> Validating...'
        print '>>> Decoding caption samples from each val batch...'
        vallosses = []
        model.eval()
        for j, (features, inputs, targets, videoids, lengths) in enumerate(valloader):
            model(features, inputs, targets, lengths, phase='val')
            valloss = model.val_loss_history[-1]
            predsent = model.captions_samples
            vallosses.append(valloss)
            if j < opt.num_val_decoding:
                for j, tup in enumerate(zip(vocab.translateSentences(targets, lengths),
                                            vocab.translateSentences(predsent, lengths))):
                    print '> tgt : {}\n> pred: {}'.format(tup[0], tup[1])
        avgvallosses = float(sum(vallosses)) / len(vallosses)
        valppl = math.exp(avgvallosses)
        print '============================================='
        print 'Epoch: {:02d}, Validation loss: {:06.2f},\tPPL: {:06.2f}'.format(i+1, avgvallosses, valppl)
        print '============================================='
        # TODO training checkpointing

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 2 every 30 epochs"""
    lr = opt.learning_rate * (0.5 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    opt.learning_rate = lr

if __name__ == '__main__':
    main()