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

        # define loss history & epoch counter
        self.loss_history = []
        self.epoch_count = 0

    def trainstep(self, imgfeatures, inputs, targets, lengths):
        targets = Variable(targets)
        if torch.cuda.is_available():
            targets = targets.cuda()

        self.optimizer.zero_grad() # reset the gradients
        outsoftmax = self.forward(imgfeatures, inputs, lengths)

        loss = self.compute_loss(outsoftmax, targets, len(lengths), max(lengths))
        loss.backward()
        if self.grad_clip > 0: # clipping to avoid exploding gradient
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()

        # log the current loss measure
        self.loss_history.append(loss.data[0])

    def compute_loss(self, outsoftmax, targets, batchsize, seqlength):
        # reshape softmax output to 2D and targets to 1D
        outsoftmax = outsoftmax.view(batchsize * seqlength, -1)
        targets = targets.view(batchsize * seqlength)
        return self.criterion(outsoftmax, targets)

    def encode_video(self, imgfeatures, lengths):
        # linear transformation of f7 vector to lower dimension
        imgfeatures = self.vid_input_fc(imgfeatures)
        # encode the sequence of frames features
        # applies sequence packing for more efficient pytorch implementation
        outvidrnn, _ = self.videoRNN(pack(imgfeatures, lengths, batch_first=True))
        outvidrnn, _ = unpack(outvidrnn, batch_first=True)
        return outvidrnn

    '''the forward function only used during the training'''
    def forward(self, imgfeatures, inputs, lengths):
        imgfeatures = Variable(imgfeatures)
        inputs = Variable(inputs)
        if torch.cuda.is_available():
            imgfeatures = imgfeatures.cuda()
            inputs = inputs.cuda()
        outvidrnn = self.encode_video(imgfeatures, lengths)

        # get word vector representation
        wordvectors = self.embedding(inputs)
        # concatenate output of first rnn with word vectors
        inpsentrnn = torch.cat((outvidrnn, wordvectors), 2) # concatenate on dim #3

        # feedforward to the sentence RNN
        outsentrnn, _ = self.sentenceRNN(pack(inpsentrnn, lengths, batch_first=True))
        outsentrnn, _ = unpack(outsentrnn, batch_first=True)

        outlogit = self.word_output_fc(outsentrnn)
        outputs = F.log_softmax(outlogit, dim=2)
        return outputs

    def forward_eval(self, imgfeatures, inputs, targets, lengths):
        with torch.no_grad():
            imgfeatures = Variable(imgfeatures)
            inputs = Variable(inputs)
            targets = Variable(targets)
            if torch.cuda.is_available():
                imgfeatures = imgfeatures.cuda()
                inputs = inputs.cuda()
            outvidrnn = self.encode_video(imgfeatures, lengths)

            # initialize the hidden vector and pass on the batch size
            hiddenvectors = self.init_hidden(len(lengths))
            # initialize first word vector which is <pad>
            wordvector = self.embedding(inputs[:,0:1])
            # iterate through the max length the RNN is unrolled
            predsent = []
            losses = []
            for i in range(max(lengths)):
                # get vector representation of a single word token
                vidvector = outvidrnn[:, i:i+1, :]
                inp = torch.cat((vidvector, wordvector), dim=2)
                out, hiddenvectors = self.sentenceRNN(inp, hiddenvectors)
                outlogit = self.word_output_fc(out)
                outsoftmax = F.log_softmax(outlogit, dim=2)
                predictedwords = outsoftmax.data.topk(1)[1][:,:,0]
                predsent.append(predictedwords)

                # update word vector for next time step
                wordvector = self.embedding(Variable(predictedwords))

                # compute single step NLL-loss
                # contiguous() to ensure the tensor located on the same memory block
                currentTarget = targets[:,i:i+1].contiguous()
                if torch.cuda.is_available():
                    currentTarget = currentTarget.cuda()
                losses.append(self.compute_loss(outsoftmax, currentTarget, len(lengths), 1))
            # return the predicted sentences and average loss over all time steps
            return torch.cat(tuple(predsent), dim=1), float(sum(losses))/len(losses)

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

    def init_hidden(self, batchsize):
        # in most recent version of pytorch the order of dimension of hidden
        # is always seq_length, batch_size, and hidden_size regardless of
        # batch_first argument values
        if self.rnn_cell.lower() == 'lstm': # tuple of h and c cells
            if torch.cuda.is_available():
                hiddenvectors = \
                    (Variable(torch.zeros(1, batchsize, self.hiddensize)).cuda(),
                     Variable(torch.zeros(1, batchsize, self.hiddensize)).cuda())
            else:
                hiddenvectors = \
                    (Variable(torch.zeros(1, batchsize, self.hiddensize)),
                     Variable(torch.zeros(1, batchsize, self.hiddensize)))
        elif self.rnn_cell.lower() == 'gru': # only h cell
            if torch.cuda.is_available():
                hiddenvectors = Variable(torch.zeros(1, batchsize, self.hiddensize)).cuda()
            else:
                hiddenvectors = Variable(torch.zeros(1, batchsize, self.hiddensize))
        return hiddenvectors