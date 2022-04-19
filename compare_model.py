"""
TextCNN model
"""

class TextCNN(nn.Module):
    def __init__(self, config, pre_weight):  # embedding_dim, drop_prob, num_filters, filter_sizes, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pre_weight)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, 256, (k, config.embedding_dim)) for k in range(2, 5)])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
      
      
 """
 Att-BiLSTM model
 """
class TextRNN_Att(nn.Module):
    def __init__(self, config, pre_weight):
        super(TextRNN_Att, self).__init__()
        self.hidden_dim = config.hidden_dim // 2
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)

        self.embeddings.weight.requires_grad = False  # 训练过程 对词向量进行微调 ?
        self.lstm = nn.LSTM(config.embedding_dim, self.hidden_dim, num_layers=config.LSTM_layers,
                            dropout=0, batch_first=True, bidirectional=True)

        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.att_w = nn.Linear(config.hidden_dim, 1, bias=True)
        self.dropout = nn.Dropout(config.dropout)

        self.fc1 = nn.Linear(config.hidden_dim, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, input, batch_seq_len, hidden=None):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        embeds = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)

        # Bi-LSTM
        if hidden is None:
            h_0, c_0 = (input.data.new(config.LSTM_layers * 2, batch_size, self.hidden_dim).fill_(0).float(),
                        input.data.new(config.LSTM_layers * 2, batch_size, self.hidden_dim).fill_(0).float())
        else:
            h_0, c_0 = hidden
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output, len_s = pad_packed_sequence(output, batch_first=True)

        # att word
        output = self.tanh1(output)
        att_ai = F.softmax(self.att_w(output), dim=1)
        output = torch.sum(output * att_ai, dim=1, keepdim=True).squeeze()
        output = self.tanh2(output)
        output = self.dropout(F.relu(self.fc1(output)))
        output = self.fc2(output)

        return output
      
      
"""
EA-Bi-LSTM model
"""

class TextAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight):
        super(TextAttention, self).__init__()
        self.hidden_dim = hidden_dim // 2
        self.embeddings = nn.Embedding.from_pretrained(pre_weight)
        self.embeddings.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=Config.LSTM_layers,
                             dropout=Config.drop_prob, batch_first=True, bidirectional=True)
        
        
        
    def forward(self, input, batch_seq_len):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        embeds_pack = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)
        # Bi-LSTM

        output, hidden = self.lstm(embeds_pack)
        output_h, len_s = pad_packed_sequence(output, batch_first=True)
#         print(output_h.size())
        
        hid_size = output_h.size(2) // 2
#         print(hid_size)
        sents_bilstm_enc = torch.cat([output_h[:, 0, :hid_size], output_h[:, -1, hid_size:]],dim=1)  # sentence bilstm output
        return output_h,len_s,sents_bilstm_enc
    
class EmojiAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo, emoji_embedding_dim):
        super(EmojiAttention, self).__init__()
        self.embeddings_emo = nn.Embedding.from_pretrained(pre_weight_emo)
        self.embeddings_emo.weight.requires_grad = False
        self.att_w = nn.Linear(hidden_dim, 1, bias=False)
        self.att_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_e = nn.Linear(emoji_embedding_dim, hidden_dim, bias=False)
        
        self.text = TextAttention(embedding_dim, hidden_dim, pre_weight)
        self.tanh = nn.Tanh()
        
    def average(self, embeds_emo, emo_len):
        # print(embeds_emo.data.shape)
        avg_embeds = torch.sum(embeds_emo.data, dim=1, keepdim=True) / emo_len
        # print(avg_embeds.shape)
        return avg_embeds
    
    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output_h, len_s, sents_bilstm_enc = self.text(input, batch_seq_len)
        _, emo_len = emos.size()
        # print(emo_len)

        embeds_emo = self.embeddings_emo(emos)
        avg_emb = self.average(embeds_emo, emo_len)
        
        """emoji-based attention word"""
        transformed_e = self.att_e(avg_emb)  # W*e
        transformed_h = self.att_h(output_h.reshape(output_h.size(0) * output_h.size(1), -1))  # W*h
       
        alpha = self.tanh(transformed_e + transformed_h.reshape(output_h.size()))
        att = self.att_w(alpha.reshape(alpha.size(0) * alpha.size(1), -1)).reshape(alpha.size(0),
                                                                                   alpha.size(1)).transpose(0, 1)
        all_att = nn.functional.softmax(att, dim=1).transpose(0, 1)  # attW,sent #ai

        output = all_att.unsqueeze(-1) * output_h  # ai_emoji*hi
        output = output.sum(1, True).squeeze()

        output = torch.cat((output, avg_emb.squeeze(), sents_bilstm_enc), dim=1)
        return output

class EmojiSentimentClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo, emoji_embedding_dim):
        super(EmojiSentimentClassification, self).__init__()
        self.emojiattention = EmojiAttention(embedding_dim, hidden_dim, pre_weight, pre_weight_emo, emoji_embedding_dim)
        self.fc1 = nn.Linear(300 * 3, 256)
        self.fc2 = nn.Linear(256, 4)  # n class
        
    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output = self.emojiattention(input, batch_seq_len, emos, batch_emo_len)
#         output = self.pool(F.relu(self.se1(self.conv1(output))))
        # output = self.pool(F.relu(self.se2(self.conv2(output))))
#         output = output.reshape(-1, 50 * 128)
        output = F.relu(self.fc1(output))
#         output = F.relu(self.fc2(output))
        output = self.fc2(output)

        return output
  
  
  """
  ECN model
  """
  
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
        self.att_h = nn.Linear(embedding_dim * 2 + hidden_dim, embedding_dim, bias=False)
        self.att_h2 = nn.Linear(hidden_dim, embedding_dim, bias=False)
        self.att_w = nn.Linear(embedding_dim * 2 + hidden_dim, 1, bias=True)

    def forward(self, input, batch_seq_len):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)
        embeds_pack = pack_padded_sequence(embeds, batch_seq_len, batch_first=True)

        output1, hidden = self.lstm1(embeds_pack)
        output_att1, len_s = pad_packed_sequence(output1, batch_first=True)
        output2, hidden = self.lstm2(output1)
        output_att2, len_s = pad_packed_sequence(output2, batch_first=True) 

        output = torch.cat((output_att1, output_att2, embeds), dim=2) 

        """W*h"""
        transformed_h = self.att_h2(output_att2)
        #reshape_h = transformed_h.reshape(output_att2.size())

        """attention word"""
        att_h = self.att_w(output)
        att_ai = nn.functional.softmax(att_h, dim=1)
        output_w = output * att_ai
        output_w = torch.sum(output_w, dim=1, keepdim=True).squeeze()  # attention后，维度900
        reshape_w = self.att_h(output_w)
        return output_att2, reshape_w, transformed_h


class EmojiAttention(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo):
        super(EmojiAttention, self).__init__()
        self.embeddings_emo = nn.Embedding.from_pretrained(pre_weight_emo) 
        self.embeddings_emo.weight.requires_grad = True

        self.tanh = nn.Tanh()
        self.att_h = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_w = nn.Linear(hidden_dim * 2, 1, bias=True)
        self.text = TextAttention(embedding_dim, hidden_dim, pre_weight)

        self.attn_word = nn.Linear(hidden_dim, hidden_dim, bias=True) 
        self.attn_e = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attn_z1 = nn.Linear(hidden_dim * 2, 1, bias=True)

        self.tanh = nn.Tanh()

    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output_h, output_w, reshape_h = self.text(input, batch_seq_len)
        _, emo_len = emos.size()
        # print(emo_len)
        embeds_emo = self.embeddings_emo(emos)
        weight_emb = self.weight(output_w, embeds_emo)


        """emoji-based attention word"""
        u = torch.cat((self.att_e(weight_emb).expand((reshape_h).size()), self.att_h(reshape_h)), dim=2)
        z = self.tanh(u)

        words_weigths = F.softmax(self.att_w(z), dim=1).permute(0, 2, 1)

        #output = all_att.unsqueeze(-1) * output_h  # ai_emoji*hi
        output = torch.bmm(words_weigths, output_h)

        output = torch.sum(output, dim=1, keepdim=True).squeeze()  # sum 3

        output = torch.cat((output.unsqueeze(1), weight_emb, output_w.unsqueeze(1)), dim=1) 
        return output

    def average(self, embeds_emo, emo_len):
        # print(embeds_emo.data.shape)
        ave_embeds = torch.sum(embeds_emo.data, dim=1, keepdim=True) / emo_len
        # print(ave_embeds.shape)
        return ave_embeds

    def weight(self, hidden, embeds_emo):
        z1 = self.tanh(
            torch.cat((self.attn_word(hidden.unsqueeze(1)).expand((embeds_emo).size()), self.attn_e(embeds_emo)),
                      dim=2))
        normalized_weigths = F.softmax(self.attn_z1(z1), dim=1).permute(0, 2, 1)
        vt = torch.bmm(normalized_weigths, embeds_emo)
        return vt

class SELayer(nn.Module):
    def __init__(self, channel, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    

class EmojiSentimentClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, pre_weight, pre_weight_emo):
        super(EmojiSentimentClassification, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=128, kernel_size=3, stride=3)
        self.pool = nn.MaxPool1d(2)
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(50 * 128, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 4)   # n class

        self.se1 = SELayer(channel=128, reduction=2)
        self.emojiattention = EmojiAttention(embedding_dim, hidden_dim, pre_weight, pre_weight_emo)

    def forward(self, input, batch_seq_len, emos, batch_emo_len, hidden=None):
        output = self.emojiattention(input, batch_seq_len, emos, batch_emo_len)
        output = self.pool(F.relu(self.se1(self.conv1(output))))
        # output = self.pool(F.relu(self.se2(self.conv2(output))))
        output = output.reshape(-1, 50 * 128)
        output = F.relu(self.fc1(output))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output
