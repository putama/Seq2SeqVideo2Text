import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

# Video-to-Sentence model
class V2S(nn.Module):
    # TODO normalization mechanism
    # TODO attentive decoder
    def __init__(self, vocabsize, featuresize=4096,
                 embedsize=500, hiddensize=1000,
                 rnn_cell='lstm'):
        super(V2S, self).__init__()
        self.vocabsize = vocabsize
        self.featuresize = featuresize
        self.embedsize = embedsize
        self.hiddensize = hiddensize

        self.embedding = nn.Embedding(self.vocabsize, self.embedsize)
        self.vid_input_fc = nn.Linear(self.featuresize, self.embedsize)

        if rnn_cell.lower() == 'lstm':
            self.videoRNN = nn.LSTM(self.embedsize, self.hiddensize,
                                    num_layers=1, batch_first=True)
            self.sentenceRNN = nn.LSTM(self.hiddensize+self.embedsize,
                                       self.hiddensize, num_layers=1, batch_first=True)
        elif rnn_cell.lower() == 'gru':
            self.videoRNN = nn.GRU(self.embedsize, self.hiddensize,
                                   num_layers=1, batch_first=True)
            self.sentenceRNN = nn.GRU(self.hiddensize + self.embedsize,
                                      self.hiddensize, num_layers=1, batch_first=True)

        self.word_output_fc = nn.Linear(self.hiddensize, self.vocabsize)
        self.init_weights()

    def forward(self, imgfeatures, targets, lengths, volatile=False):
        batchsize = len(lengths)
        imgfeatures = Variable(imgfeatures, volatile=volatile)
        targets = Variable(targets, volatile=volatile)

        # linear transformation of f7 vector to lower dimension
        imgfeatures = imgfeatures.view(-1, self.featuresize) # reshape to a matrix
        imgfeatures = self.vid_input_fc(imgfeatures)
        imgfeatures = imgfeatures.view(batchsize, -1, self.embedsize) # reshape back to tensor

        # encode the sequence of frames features
        # applies sequence packing for more efficient pytorch implementation
        outvidrnn, _ = self.videoRNN(pack(imgfeatures, lengths, batch_first=True))
        outvidrnn, _ = unpack(outvidrnn, batch_first=True)

        # get word vector representation
        wordvectors = self.embedding(targets)

        # concatenate output of first rnn with word vectors
        inpsentrnn = torch.cat((outvidrnn, wordvectors), 2) # concatenate on dim #3

        # feedforward to the sentence RNN
        outsentrnn, _ = self.sentenceRNN(pack(inpsentrnn, lengths, batch_first=True))
        outsentrnn, _ = unpack(outsentrnn, batch_first=True)

        return imgfeatures, targets, outvidrnn

    def init_weights(self):
        """Xavier initialization for the fully connected networks"""
        # embedding linear transformation
        r = np.sqrt(6.) / np.sqrt(self.vid_input_fc.in_features +
                                  self.vid_input_fc.out_features)
        self.vid_input_fc.weight.data.uniform_(-r, r)
        self.vid_input_fc.bias.data.fill_(0)
        # logit layer transformation
        r = np.sqrt(6.) / np.sqrt(self.word_output_fc.in_features +
                                  self.word_output_fc.out_features)
        self.word_output_fc.weight.data.uniform_(-r, r)
        self.word_output_fc.bias.data.fill_(0)