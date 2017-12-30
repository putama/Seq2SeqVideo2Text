import torch
import torch.utils.data as data
import numpy as np

class MSVDPrecompDataset(data.Dataset):
    """
    Load precomputed features of video frames
    of MSVD video text dataset
    """
    def __init__(self, data_path, data_split, vocab, opt):
        self.opt = opt
        self.vocab = vocab
        self.data_path = data_path+'msvd/'
        self.features_path = self.data_path+'features_precomputed/yt_allframes_vgg_fc7_'+data_split+'.txt'
        self.captions_path = self.data_path+'captions/sents_'+data_split+'_lc_nopunc.txt'
        print 'load video features...'
        self.videoid2frames = self.txt2videosfeat(self.features_path)
        print 'load video captions...'
        self.captiontuples = self.txt2captions(self.captions_path)

        # assert whether videos in captions and frames features match
        assert set(self.videoid2frames.keys()) == set(map(lambda x: x[0], self.captiontuples))

        self.length = len(self.captiontuples)

    def __getitem__(self, index):
        videoid = self.captiontuples[index][0]
        captionraw = self.captiontuples[index][1]
        inpcapraw = captionraw[0:-1]
        targetcapraw = captionraw[1:]
        featuresraw = self.videoid2frames[videoid]

        # trim frames in case total length exceed opt.seq_maxlen
        if len(featuresraw) + len(inpcapraw) > self.opt.seq_maxlen:
            print 'exceeding!'
            exceednum = self.opt.seq_maxlen - (len(featuresraw) + len(inpcapraw))
            featuresraw = featuresraw[0:-exceednum]

        return featuresraw, inpcapraw, targetcapraw, videoid

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
                caption = []
                caption.append(self.vocab('<start>'))
                caption.extend(map(lambda x: self.vocab(x), captionstr.split()))
                caption.append(self.vocab('<end>'))
                captiontuples.append((videoid, caption))
        return captiontuples

    def __len__(self):
        return self.length

def collate_precomputed(data):
    '''Build mini-batch tensors of captions and features list '''
    # sorted by caption length for later pad packed sequence
    data.sort(key=lambda x: len(x[0])+len(x[1]), reverse=True)
    featuresraws, inpcapraws, targetcapraws, videoids = zip(*data)

    # forward and backward padding of caption sequences
    lengths = [len(i[0])+len(i[1]) for i in data]
    inpcaptions = torch.zeros(len(inpcapraws), max(lengths)).long()
    for i, cap in enumerate(inpcapraws):
        start = len(featuresraws[i])
        end = lengths[i]
        inpcaptions[i, start:end] = torch.Tensor(cap)
    targetcaptions = torch.zeros(len(targetcapraws), max(lengths)).long()
    for i, cap in enumerate(targetcapraws):
        start = len(featuresraws[i])
        end = lengths[i]
        targetcaptions[i, start:end] = torch.Tensor(cap)

    # forward padding of frame features
    features = torch.zeros(len(featuresraws), max(lengths), len(featuresraws[0][0]))
    for i, feat in enumerate(featuresraws):
        start = 0
        end = len(featuresraws[i])
        features[i, start:end, :] = torch.Tensor(featuresraws[i]).unsqueeze(0)

    return features, inpcaptions, targetcaptions, videoids, lengths
