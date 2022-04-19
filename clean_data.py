"""
clean data
"""


import pandas as pd
import numpy as np
import emoji
import re
import csv

def find_all_emoji(str2):
    emojis_uni = []
    emojis_text = []
    elist = re.findall(':[\w-]+?:', str2)
    return elist


def filter_emoji(desstr, restr=''):

    return re.sub(':\S+?:', restr, desstr)
# cnt = 0
text = []
emojis_text = []
labels = []
# label_dict = {'neutral': 1, 'positive': 2, 'negative': 0}
label_dict = {'0': 0, '1': 1, '2': 2}
# read csv file line by line
with open('C:/Users/yxw-thinkpad-e470c/Desktop/generic_sentiment_dataset_10k.csv','r',encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    header_row = next(reader)
    datas = []
    for row in reader:
        # print(row)
        if len(row) < 2:
            continue
        line = row[1]
        line = re.sub('\n', ' ', line)
        line = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', "", line)
        # filter http
        line = re.sub(r"[\w]+(\.[\w]+)*@[\w]+(\.[\w])+", "", line)  # filter mail
        symbol = "[_.!+-=——,$%^，。？、~@#￥%……&*《》<>「」{}【】()/\\\[\]'\"]"
        line = re.sub(symbol, ' ', line)
        line = re.sub(r"\s+", " ", line)  # merge space
        label = row[2]
        emoji_text = emoji.demojize(line)
        clean_text = filter_emoji(emoji_text)
        flag = 0  # include emoji or not
        if clean_text:  # include text or not
            all = find_all_emoji(emoji_text)

            if all:

                for each in all:
                    if each != emoji.emojize(each):
                        flag = 1
                        break
        if flag==1:
            text.append(line)
            emojis_text.append(emoji_text)
            labels.append(label_dict[label])
            # print(line)
                # print(emoji.emojize(each))
# data = pd.read_csv('F:/sentiment140/testdata.manual.2009.06.14.csv', sep=',')
# print(cnt)

# # data = np.array(data)
# col_1 = data.iloc[0]
# print(col_1)
# col_2 = data.iloc[5]
# labels = np.array(col_1)
# for item in col_2:
#     es = emoji.demojize(item)
#     all = find_all_emoji(es)
#     if all:
#         print(all)
# print(col_1)
with open('../data/generate/corpus_10k.txt', 'wb') as f:
    for i in range(len(text)):
        line = text[i]
        la = labels[i]
        line = str(la) + '\t' + line + '\n'
        f.write(line.encode('utf-8'))


"""compare corpus data"""
with open('../data/compare/corpus_10k.txt', 'wb') as f:
    for i in range(len(emojis_text)):
        line = emojis_text[i]
        la = labels[i]
        line = str(la) + '\t' + line + '\n'
        f.write(line.encode('utf-8'))
