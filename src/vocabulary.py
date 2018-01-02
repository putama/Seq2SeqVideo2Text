class Vocabulary(object):
    '''
    Simple vocabulary wrapper class
    Inspired by Faghri (2017)
    '''
    def __init__(self, vocab_path):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0
        self.add_word('<pad>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.build_vocab(vocab_path)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocab(self, vocab_path):
        with open(vocab_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.add_word(line.split('\n')[0])

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<en_unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def getWord(self, idx):
        if idx not in self.idx2word:
            return self.idx2word[self.word2idx['<en_unk>']]
        else:
            return self.idx2word[idx]

    def getPadding(self):
        return self.word2idx['<pad>']

    def translateSentences(self, sentences, lengths):
        translated = [''] * len(lengths)
        for k in range(max(lengths)):
            for m in range(len(sentences)):
                word1 = self.getWord(sentences[m,k])
                if not word1 in ['<pad>']:
                    translated[m] = translated[m]+word1+' '
                else:
                    translated[m] = translated[m]+'+ '
        return translated