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
parser.add_argument('--print_freq', default=50)
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

dataset = data.MSVDPrecompDataset(data_path, 'training', vocab, opt)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=opt.batch_size,
                                         shuffle=True,
                                         num_workers=4,
                                         collate_fn=data.collate_precomputed)

# initialize the model
model = V2S(opt)
model.train()
for i in range(opt.epoch):
    start = time.time()
    for j, (features, inputs, targets, videoids, lengths) in enumerate(dataloader):
        model.trainstep(features, inputs, targets, lengths)
        # prints training progression
        if (j+1) % opt.print_freq == 0:
            losses = model.loss_history[-opt.print_freq:-1]
            avglosses = float(sum(losses)) / len(losses)
            ppl = math.exp(avglosses)
            print 'Epoch: {}, Iter: {}/{}, Current loss: {}, PPL: {}, Epoch time elapsed: {}s'.format(
                str(i+1), str(j+1), str(len(dataloader)), avglosses, ppl, str(time.time()-start)
            )
            predsent, evalloss = model.forward_eval(features, inputs, targets, lengths)
    # TODO training checkpointing
    # TODO evaluating current model