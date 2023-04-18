import argparse
import pandas as pd
import torch
import torchtext.data as data
from torchtext.vocab import Vectors
import joblib
import re
import json

import model
import train


parser = argparse.ArgumentParser(description='TextCNN text classifier')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 128]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stopping', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embedding-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-filter-num', type=int, default=100, help='number of each size of filter')
parser.add_argument('-filter-sizes', type=str, default='3,4,5',
                    help='comma-separated filter sizes to use for convolution')

parser.add_argument('-static', type=bool, default=True, help='whether to use static pre-trained word vectors')
parser.add_argument('-non-static', type=bool, default=False, help='whether to fine-tune static pre-trained word vectors')
parser.add_argument('-multichannel', type=bool, default=False, help='whether to use 2 channel of word vectors')
parser.add_argument('-pretrained-name', type=str, default='sgns.weibo.word',
                    help='filename of pre-trained word vectors')
parser.add_argument('-pretrained-path', type=str, default='pretrained', help='path of pre-trained word vectors')

# device
parser.add_argument('-device', type=int, default=0, help='device to use for iterate data, -1 mean cpu [default: -1]')

# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
args = parser.parse_args()

parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
# parser.add_argument('-test', action='store_true', default=False, help='train or test')

# load vocab
print('\nLoading vocab...\n')
text_field = joblib.load('vocab\\text_field.pt') 
label_field = joblib.load('vocab\\label_field.pt') 
print('\nfinishing...\n')


args.vocabulary_size = len(text_field.vocab)
if args.static:
    args.embedding_dim = text_field.vocab.vectors.size()[-1]
    args.vectors = text_field.vocab.vectors
if args.multichannel:
    args.static = True
    args.non_static = True
args.class_num = len(label_field.vocab)
args.cuda = args.device != -1 and torch.cuda.is_available()
args.filter_sizes = [int(size) for size in args.filter_sizes.split(',')]


print('Parameters:')
for attr, value in sorted(args.__dict__.items()):
    if attr in {'vectors'}:
        continue
    print('\t{}={}'.format(attr.upper(), value))


# load model
args.snapshot = "snapshot\\best_steps_2400.pt"
text_cnn = model.TextCNN(args)
print('\nLoading model from {}...\n'.format(args.snapshot))
text_cnn.load_state_dict(torch.load(args.snapshot))



if args.cuda:
    torch.cuda.set_device(args.device)
    text_cnn = text_cnn.cuda()


regex = re.compile(r'#[^#]+#')

file = open("Tweets.json",encoding="utf-8")
k = 0
# num1 = 0
# num2 = 0
# num3 = 0
data = json.load(file)
for i in data:
    s = i["content"]
    label = -2
    args.predict = s
    args.predict = regex.sub("",args.predict)
    try:
        if args.predict is not None and args.predict != "":
            label = train.predict(args.predict, text_cnn, text_field, label_field, args.cuda)
    except Exception as e:
        print(e)
        
    data[k]['label'] = label
    k+=1
    # if label == '0':
    #     num1 += 1
    # else label == '1':
    #     num2 += 1

re = open("label.json",'w',encoding="utf-8")
json.dump(data,re,ensure_ascii=False)


