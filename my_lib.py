import re
import Stemmer
import numpy as np
import codecs

def spec2space(text):
    chars = "\\`*_{}[]()>#+-.!|,$@?"
    for c in chars:
        if c in text:
            text = text.replace(c, " ")
    return text


def splitstringStem(str):
    words = []
    str = str.lower()
    str = str.replace("ё", "е")
    stemmer = Stemmer.Stemmer('russian')
    # for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,\d,!,?,"]',str):
    # re.split("(?:(?:[^а-яА-Я]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)"
    for i in re.split("(?:[^а-я0-9]+)", str):
        if len(i) > 1 and len(i) <= 17:
            words.append(stemmer.stemWord(i))
            # words.append(i) # without stamming
    return words

def splitstringStemKaz(str):
    words = []
    str = str.lower()
    str = str.replace("ё", "е")
    stemmer = Stemmer.Stemmer('russian')
    # for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,\d,!,?,"]',str):
    # re.split("(?:(?:[^а-яА-Я]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)"
    for i in re.split("(?:[^а-я0-9әғқңөұүһі]+)", str):
        if len(i) > 1 and len(i) <= 17:
            stemmed=stemmer.stemWord(i)
            if len(stemmed)>1:
                words.append(stemmed)
            # words.append(i) # without stamming
    return words

def splitstringStemKazStop(str, stoplist):
    words = []
    str = str.lower()
    str = str.replace("ё", "е")
    stemmer = Stemmer.Stemmer('russian')
    # for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,\d,!,?,"]',str):
    # re.split("(?:(?:[^а-яА-Я]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)"
    for i in re.split("(?:[^а-я0-9әғқңөұүһі]+)", str):
        if len(i) > 1 and len(i) <= 17:
            if i not in stoplist:
                stemmed=stemmer.stemWord(i)
                if len(stemmed)>1:
                    words.append(stemmed)
                    # words.append(i) # without stamming

    return words

def splitstring(str):
    words = []
    str = str.lower()
    str = str.replace("ё", "е")
    # for i in re.split('[;,.,\n,\s,:,-,+,(,),=,/,«,»,\d,!,?,"]',str):
    # re.split("(?:(?:[^а-яА-Я]+')|(?:'[^a-zA-Z]+))|(?:[^a-zA-Z']+)"
    for i in re.split("(?:[^а-я0-9]+)", str):
        if len(i) > 1 and len(i) <= 17:
            words.append(i)
            # words.append(i) # without stamming
    return words

def WordReplace(dataset, dict):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in dict:
                dataset[i][j] = dict[dataset[i][j]]
            else:
                dataset[i][j] = 2  # unknown
    return dataset


def WordReplaceNP(data, dict):
    a=np.zeros((len(data),len(max(data, key=len))), dtype=np.uint16)
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] in dict:
                a[i][j] = dict[data[i][j]]
            else:
                a[i][j] = 2  # unknown
    return a


# convert array to sentence. just for tests
def WordReturn(dataset, dict):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in dict:
                print(dict[dataset[i][j]], end=' ')
            else:
                dataset[i][j] = 2  # unknown
        print()


def WordsDic(dataset):
    word = {"<PAD>": 0, "<START>": 1, "<UNK>": 2, "<UNUSED>": 3}
    index = 4
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if dataset[i][j] in word:
                None
            else:
                word.update({dataset[i][j]: index})
                index += 1
    return word
