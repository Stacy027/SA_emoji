"""
hybrid-attention network
"""

import re
import random
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torch.nn.functional as F
import gensim


class DictObj(object):
    def __init__(self, mp):
        self.map = mp

    def __setattr__(self, name, value):
        if name == 'map':
            object.__setattr__(self, name, value)
            return

        self.map[name] = value

    def __getattr__(self, name):
        return self.map[name]


Config = DictObj({
    'train_path': "./corpus_sentient_train_40.txt",
    'test_path': "./corpus_sentiment_test.txt",
    'pred_word2vec_path': './GoogleNews-vectors-negative300.bin.gz',  # word embeddings
    'pred_emo2vec_path': './emo2vec_tf.txt',  # emoji embeddings
    'word_embedding_dim': 300,
    'emoji_embedding_dim': 300,
    'hidden_dim': 300,  # LSTM
    'lr': 0.001,
    'LSTM_layers': 1,
    'drop_prob': 0,
    'seed': 0
})


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt

def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    rtn = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        rtn.append(correct_k.mul_(100.0 / batch_size))
    return rtn



def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True



def build_word_dict(train_path):
    words = []
    emojis = []
    max_len = 0
    total_len = 0
    with open(train_path, 'r', encoding='UTF-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            word = re.split(r'[\t]', line)[1].split(' ')
            emo = re.split(r'[\t]', line)[-1].split(' ')
            #             print(word)
            max_len = max(max_len, len(word))
            total_len += len(word)
            for w in word:
                words.append(w)
            for e in emo:
                emojis.append(e)
    words = list(set(words))  # remove duplication
    words = sorted(words)  # sort

    emojis = list(set(emojis))
    emojis = sorted(emojis)

    word2ix = {w: i + 1 for i, w in enumerate(words)}
    ix2word = {i + 1: w for i, w in enumerate(words)}
    word2ix['<unk>'] = 0
    ix2word[0] = '<unk>'
    avg_len = total_len / len(lines)
    emo2ix = {w: i + 1 for i, w in enumerate(emojis)}
    ix2emo = {i + 1: w for i, w in enumerate(emojis)}
    emo2ix['<unk>'] = 0
    ix2emo[0] = '<unk>'
    return word2ix, ix2word, max_len, avg_len, emo2ix, ix2emo


class CommentDataSet(Dataset):
    def __init__(self, data_path, word2ix, ix2word, emo2ix, ix2emo):
        self.data_path = data_path
        self.word2ix = word2ix
        self.ix2word = ix2word
        self.emo2ix = emo2ix
        self.ix2emo = ix2emo
        self.data, self.label, self.emo = self.get_data_label()

    def __getitem__(self, idx: int):
        return self.data[idx], self.label[idx], self.emo[idx]

    def __len__(self):
        return len(self.data)

    def get_data_label(self):
        text = []
        emo = []
        label = []
        count = 0
        with open(self.data_path, 'r', encoding='UTF-8') as f:
            #             lines = f.readlines()
            for line in f:
                line = line.strip().split("\t")
                count += 1
                try:
                    if (len(line) > 2):  # include text or not
                        line_words = line[1].strip().split(" ")
                        line_emos = line[2].strip().split(" ")
                    else:
                        line_emos = line[1].strip().split(" ")
                    label.append(torch.tensor(int(line[0]), dtype=torch.int64))

                except BaseException:
                    print('not expected line:' + line + str(count))
                    continue

                words_to_idx = []
                for w in line_words:
                    try:
                        index = self.word2ix[w]
                    except BaseException:
                        index = 0
                    words_to_idx.append(index)
                emos_to_idx = []
                for e in line_emos:
                    try:
                        index2 = self.emo2ix[e]
                    except BaseException:
                        index2 = 0
                    emos_to_idx.append(index2)

                text.append((torch.tensor(words_to_idx, dtype=torch.int64)))
                emo.append((torch.tensor(emos_to_idx, dtype=torch.int64)))
                # data = [text, emo]
        return text, label, emo


def mycollate_fn(data):
    # get （input，label, emo）triple in batch_size
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [len(sq[0]) for sq in data]
    input_data = []
    label_data = []
    emo_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])
        emo_data.append(i[2])
    input_data = pad_sequence(input_data, batch_first=True, padding_value=0)
    label_data = torch.tensor(label_data)
    emo_data = pad_sequence(emo_data, batch_first=True, padding_value=0)

    # data.sort(key=lambda x: len(x[2]), reverse=True)
    emo_length = [len(eq[2]) for eq in data]
    return input_data, label_data, emo_data, data_length, emo_length


def pre_emoji_weight(vocab_size):
    cnt = 0
    weight = torch.zeros(vocab_size, Config.emoji_embedding_dim)
    # initial weights
    for i in range(len(emo2vec_model.index_to_key)):
        try:
            index = emo2ix[emo2vec_model.index_to_key[i]]
        except:
            continue

        ix = index
        weight[ix, :] = torch.from_numpy(emo2vec_model.get_vector(
            emo2vec_model.index_to_key[i]))
        cnt += 1
    print(cnt)
    return weight


def pre_weight(vocab_size):
    cnt = 0
    weight = torch.zeros(vocab_size, Config.word_embedding_dim)
    # initial weights
    for i in range(len(word2vec_model.index_to_key)):
        try:
            index = word2ix[word2vec_model.index_to_key[i]]
        except:
            continue

        ix = index
        weight[ix, :] = torch.from_numpy(word2vec_model.get_vector(
            word2vec_model.index_to_key[i]))
        cnt += 1
    print(cnt)
    return weight


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class TextAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight):
        super(TextAttention, self).__init__()
        self.hidden_dim = hidden_dim // 2
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        self.embeddings.weight.requires_grad = False
        self.lstm1 = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                             dropout=Config.drop_prob, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                             dropout=Config.drop_prob, batch_first=True, bidirectional=True)

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # after self-attention, a full-connected layer with LayerNorm
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)

    def forward(self, input, batch_seq_len):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        embeds_pack = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
        # Bi-LSTM
        output1, hidden = self.lstm1(embeds_pack)
        output_att1, len_s = pad_packed_sequence(output1, batch_first=True)
        output2, hidden = self.lstm2(output1)
        output_att2, len_s = pad_packed_sequence(output2, batch_first=True)  # hidden layer
        """self-attention"""
        # generate QKV matrices
        Q = self.Q_linear(output_att2)
        K = self.K_linear(output_att2).permute(0, 2, 1)  # transpose
        V = self.V_linear(output_att2)

        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=2)
        self_att = torch.matmul(alpha, V)
        return output_att2, self_att


class EmojiAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo):
        super(EmojiAttention, self).__init__()
        self.embeddings_emo = nn.Embedding.from_pretrained(pre_weight_emo)  # pre-trained
        #         self.embeddings_emo = nn.Embedding(len(ix2emo), 300)  # random
        #         self.embeddings_emo = nn.Embedding.from_pretrained(torch.zeros(len(ix2emo), 300)）  # 0

        self.embeddings_emo.weight.requires_grad = True
        self.text = TextAttention(embedding_dim, hidden_dim, pre_weight)
        #         self.tanh = nn.Tanh()

        """v_e= CA(u,e,e)"""
        self.Q1 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.K1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.V1 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        """v_h= CA(e,h,h)"""
        self.Q2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.K2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.V2 = nn.Linear(hidden_dim, embedding_dim, bias=False)

        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(embedding_dim, embedding_dim)

        self.LayerNorm = LayerNorm(hidden_dim, eps=1e-12)
        self.out_dropout = nn.Dropout(0.5)


    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output_h, output_u = self.text(input, batch_seq_len)
        _, emo_len = emos.size()

        embeds_emo = self.embeddings_emo(emos)

        weight_emoji = self.co_attention_h(output_h, embeds_emo)
        weight_text = self.co_attention_e(output_h, embeds_emo)
        return output_u, weight_emoji, weight_text

    def average(self, embeds_emo, emo_len):
        # print(embeds_emo.data.shape)
        ave_embeds = torch.sum(embeds_emo.data, dim=1, keepdim=True) / emo_len
        # print(ave_embeds.shape)
        return ave_embeds

    def co_attention_e(self, hidden, embeds_emo):
        # generate QKV matrices
        Q = self.Q1(hidden)
        K = self.K1(embeds_emo).permute(0, 2, 1)  # transpose
        V = self.V1(embeds_emo)

        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=2)
        eco_att = torch.matmul(alpha, V)
        return eco_att

    def co_attention_h(self, hidden, embeds_emo):
        # generate QKV matrices
        Q = self.Q2(embeds_emo)
        K = self.K2(hidden).permute(0, 2, 1)  # transpose
        V = self.V2(hidden)

        alpha = torch.matmul(Q, K)
        alpha = F.softmax(alpha, dim=2)
        hco_att = torch.matmul(alpha, V)
        return hco_att


class EmojiSentimentClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo):
        super(EmojiSentimentClassification, self).__init__()
        self.filter_sizes = (2, 1, 3)
        self.num_filters = 128  # channels
        self.num_classes = 4
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, embedding_dim)) for k in self.filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)
        self.emojiattention = EmojiAttention(embedding_dim, hidden_dim, pre_weight, pre_weight_emo)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output_u, weight_emoji, weight_text = self.emojiattention(input, batch_seq_len, emos, batch_emo_len)
        out = [output_u.unsqueeze(1), weight_emoji.unsqueeze(1), weight_text.unsqueeze(1)]
        output = torch.cat([self.conv_and_pool(out[i], self.convs[i]) for i in range(len(out))], 1)
        output = self.dropout(output)
        output = self.fc(output)

        return output


def train(epoch, epochs, train_loader, device, model, criterion, optimizer, scheduler):
    model.train()
    top1 = AvgrageMeter()
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):  # default 0
        inputs, labels, emos, batch_seq_len, batch_emo_len = data[0].to(device), data[1].to(device), data[2].to(device), \
                                                             data[3], data[4]

        optimizer.zero_grad()
        outputs = model(inputs, batch_seq_len, emos, batch_emo_len)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, pred = outputs.topk(1)
        prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
        n = inputs.size(0)
        top1.update(prec1.item(), n)

        train_loss += loss.item()
        postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.6f' % top1.avg}
        train_loader.set_postfix(log=postfix)
    scheduler.step()


def test(epoch, validate_loader, device, model, criterion):
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        val_top1 = AvgrageMeter()
        validate_loss = 0.0
        validate_loader = tqdm(validate_loader)
        for i, data in enumerate(validate_loader, 0):
            inputs, labels, emos, batch_seq_len, batch_emo_len = data[0].to(device), data[1].to(device), data[2].to(
                device), data[3], data[4]
            outputs = model(inputs, batch_seq_len, emos, batch_emo_len)
            loss = criterion(outputs, labels)
            prec1, prec2 = accuracy(outputs, labels, topk=(1, 2))
            n = inputs.size(0)
            val_top1.update(prec1.item(), n)

            validate_loss += loss.item()
            postfix = {'validate_loss': '%.6f' % (validate_loss / (i + 1)), 'validate_acc': '%.6f' % val_top1.avg}
            validate_loader.set_postfix(log=postfix)
        val_acc = val_top1.avg
    return val_acc


word2ix, ix2word, max_len, avg_len, emo2ix, ix2emo = build_word_dict(Config.train_path)
train_data = CommentDataSet(Config.train_path, word2ix, ix2word, emo2ix, ix2emo)
# data, label, emo = train_data.get_data_label()
train_loader = DataLoader(train_data, batch_size=16, shuffle=True,
                          num_workers=0, collate_fn=mycollate_fn)

test_data = CommentDataSet(Config.test_path, word2ix, ix2word, emo2ix, ix2emo)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False,
                         num_workers=0, collate_fn=mycollate_fn)

word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(Config.pred_word2vec_path, binary=True)  # google
emo2vec_model = gensim.models.KeyedVectors.load_word2vec_format(Config.pred_emo2vec_path, binary=False)

w_features = pre_weight(len(word2ix))
e_features = pre_emoji_weight(len(ix2emo))

model = EmojiSentimentClassification(embedding_dim=Config.word_embedding_dim,
                                    hidden_dim=Config.hidden_dim,
                                    pre_weight=w_features,
                                    pre_weight_emo=e_features)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(model)
epochs = 10
optimizer = optim.Adam(model.parameters(), lr=Config.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # study ratio
criterion = nn.CrossEntropyLoss()
# model.load_state_dict(torch.load(Config.save_model_path, map_location=torch.device('cpu')))

for epoch in range(epochs):
    train_loader = tqdm(train_loader)

    print('[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, epochs, 'lr:', scheduler.get_lr()[0]))
    train(epoch, epochs, train_loader, device, model, criterion, optimizer, scheduler)
    test(epoch, test_loader, device, model, criterion)
