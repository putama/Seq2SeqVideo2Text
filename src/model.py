import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.clip_grad import clip_grad_norm

# Video-to-Sentence model
class V2S(nn.Module):
    # TODO word padding uses zero vectors instead
    # TODO normalization mechanism
    # TODO attentive decoder
    # TODO curriculum learning / teacher forcing
    def __init__(self, opt):
        super(V2S, self).__init__()
        self.rnn_cell = opt.rnn_cell
        self.vocabsize = opt.vocab_size
        self.featuresize = opt.feature_size
        self.embedsize = opt.embedding_size
        self.hiddensize = opt.hidden_size
        self.grad_clip = opt.grad_clip

        self.embedding = nn.Embedding(self.vocabsize, self.embedsize)
        self.vid_input_fc = nn.Linear(self.featuresize, self.embedsize)

        if self.rnn_cell.lower() == 'lstm':
            self.videoRNN = nn.LSTM(self.embedsize, self.hiddensize,
                                    num_layers=1, batch_first=True)
            self.sentenceRNN = nn.LSTM(self.hiddensize+self.embedsize,
                                       self.hiddensize, num_layers=1, batch_first=True)
        elif self.rnn_cell.lower() == 'gru':
            self.videoRNN = nn.GRU(self.embedsize, self.hiddensize,
                                   num_layers=1, batch_first=True)
            self.sentenceRNN = nn.GRU(self.hiddensize + self.embedsize,
                                      self.hiddensize, num_layers=1, batch_first=True)

        self.word_output_fc = nn.Linear(self.hiddensize, self.vocabsize)
        self.init_weights()

        # define the criterion and optimizer
        self.criterion = nn.NLLLoss()
        self.params = self.parameters()
        self.optimizer = torch.optim.Adam(self.params, opt.learning_rate)

        if torch.cuda.is_available():
            self.cuda()

        # define loss history
        self.loss_history = []

    def trainstep(self, imgfeatures, inputs, targets, lengths):
        targets = Variable(targets)
        if torch.cuda.is_available():
            targets = targets.cuda()

        self.optimizer.zero_grad() # reset the gradients
        outsoftmax = self.forward(imgfeatures, inputs, lengths)

        # reshape softmax output to 2D and targets to 1D
        outsoftmax = outsoftmax.view(len(lengths) * max(lengths), -1)
        targets = targets.view(len(lengths) * max(lengths))

        loss = self.criterion(outsoftmax, targets)
        loss.backward()
        if self.grad_clip > 0: # clipping to avoid exploding gradient
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        # log the current loss measure
        self.loss_history.append(loss.data[0])

    def forward(self, imgfeatures, inputs, lengths, volatile=False):
        imgfeatures = Variable(imgfeatures, volatile=volatile)
        inputs = Variable(inputs, volatile=volatile)

        if torch.cuda.is_available():
            imgfeatures = imgfeatures.cuda()
            inputs = inputs.cuda()

        # linear transformation of f7 vector to lower dimension
        imgfeatures = self.vid_input_fc(imgfeatures)

        # encode the sequence of frames features
        # applies sequence packing for more efficient pytorch implementation
        outvidrnn, _ = self.videoRNN(pack(imgfeatures, lengths, batch_first=True))
        outvidrnn, _ = unpack(outvidrnn, batch_first=True)

        # get word vector representation
        wordvectors = self.embedding(inputs)

        # concatenate output of first rnn with word vectors
        inpsentrnn = torch.cat((outvidrnn, wordvectors), 2) # concatenate on dim #3

        # feedforward to the sentence RNN
        outsentrnn, _ = self.sentenceRNN(pack(inpsentrnn, lengths, batch_first=True))
        outsentrnn, _ = unpack(outsentrnn, batch_first=True)

        outlogit = self.word_output_fc(outsentrnn)
        output = F.log_softmax(outlogit, dim=2)
        return output

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
        # word embedding weights initialization
        self.embedding.weight.data.uniform_(-0.08, 0.08)