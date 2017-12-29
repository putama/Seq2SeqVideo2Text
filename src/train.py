import argparse
import torch
import data
from vocabulary import Vocabulary
from model import V2S

parser = argparse.ArgumentParser()
parser.add_argument('--seq_maxlen', default=80)
opt = parser.parse_args()

data_path = '../data/'
vocab_path = data_path+'msvd/coco_vocabulary.txt'
vocab = Vocabulary(vocab_path)
dataset = data.MSVDPrecompDataset(data_path, 'debug', vocab, opt)
dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=5,
                                         shuffle=True,
                                         num_workers=2,
                                         collate_fn=data.collate_precomputed)

# initialize the model
model = V2S(vocabsize=len(vocab))

for i, (features, captions, videoids, lengths) in enumerate(dataloader):
    print i
    print videoids
    model(features, captions, lengths)