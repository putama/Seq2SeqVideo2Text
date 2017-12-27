import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from vocabulary import Vocabulary

class MSVDPrecompDataset(data.Dataset):
    """
    Load precomputed features of video frames
    of MSVD video text dataset
    """
    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        self.data_path = data_path+'msvd/'
        self.features_path = self.data_path+'features_precomputed/yt_allframes_vgg_fc7_'+data_split+'.txt'
        self.captions_path = self.data_path+'captions/sents_'+data_split+'_lc_nopunc.txt'
        print 'load video features...'
        self.videoid2frames = self.txt2videosfeat(self.features_path)
        print 'load video captions...'
        self.captiontuples = self.txt2captions(self.captions_path)
        print len(self.videoid2frames)

    def txt2videosfeat(self, features_path):
        videoid2frames = {}
        with open(features_path, 'r') as f:
            lines = f.readlines()
            videocounter = 0
            for line in lines:
                splits = line.split(',')
                videoid = splits[0].split('_')[0]
                frameid = splits[0].split('_')[2]
                featvector = np.array(map(lambda x: float(x), splits[1:]))

                if not videoid2frames.has_key(videoid):
                    videoid2frames[videoid] = []
                    # log how many videos loaded so far
                    videocounter += 1
                    if videocounter % 50 == 0: print '{} videos loaded.'.format(str(videocounter))

                videoid2frames[videoid].append(featvector)
                # make sure that the frames are stored chronologically ordered
                assert len(videoid2frames[videoid]) == int(frameid)
        return videoid2frames

    def txt2captions(self, captions_path):
        captiontuples = []
        with open(captions_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                splits = line.split('\t')
                videoid = splits[0]
                captionstr = splits[1]
                caption = map(lambda x: self.vocab(x), captionstr.split())
                captiontuples.append((videoid, caption))
        return captiontuples

data_path = '../data/'
vocab_path = data_path+'msvd/coco_vocabulary.txt'
vocab = Vocabulary(vocab_path)
dataset = MSVDPrecompDataset(data_path, 'val', vocab)