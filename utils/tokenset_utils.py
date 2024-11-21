import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import OrderedDict
from typing import List

regexStripID = re.compile(r" \(\d+\)")
regexGetID = re.compile(r"\s? \((\d+)\)\s?\> ")
regexSplit = re.compile(r" > ")

def normalise_breadcrumb(text, link = 'raw'):
    if link == 'symbol':
        segments = re.split(regexSplit, text + ' > ')[:-1]
        crumb = re.split(regexStripID, segments[0])[0]
        if len(segments) > 1:
            for i in range(1, len(segments)):
                crumb += ' > '
                crumb += re.split(regexStripID, segments[i])[0]
    elif link == 'word':
        segments = re.split(regexSplit, text + ' > ')[:-1]
        crumb = re.split(regexStripID, segments[0])[0]
        if len(segments) > 1:
            for i in range(1, len(segments)):
                if i == 1:
                    crumb += ' contains '
                else:
                    crumb += ', which contains '
                crumb += re.split(regexStripID, segments[i])[0]
        crumb += '.'
    elif link == 'id':
        segments = re.findall(regexGetID, text + ' > ')
        crumb = segments[0]
        if len(segments) > 1:
            for i in range(1, len(segments)):
                crumb += ' '
                crumb += segments[i]
    else:
        crumb = text
    return crumb

def segment(breadcrumb:str, sep:str):
    return breadcrumb.split(' ' + sep + ' ')

def listunion(x,y):
    return list(OrderedDict.fromkeys(x + y))

def listdiff(x,y):
    return [z for z in x if z not in y]

def listadd(l,x):
    return l + [x] if x not in l else l

def tokenise(s:str or List[str]):
    if s == []:
        return []
    if type(s) == str:
        return word_tokenize(s)
    return word_tokenize(s[0]) + tokenise(s[1:])

def lemma(word,pos,use_pos=True):
    wnl = WordNetLemmatizer()
    return wnl.lemmatize(word,pos) if use_pos else wnl.lemmatize(word)

def tokenset(text,lemmatize=True,use_pos=False):
    pos_replace = {'NOUN':'n','VERB':'v','ADJ':'a','ADV':'r','NUM':'n'}
    segments = segment(text,sep='>')
    tokens = []
    for seg in segments:
        segtokens = tokenise(seg.lower())
        tagged = nltk.pos_tag(segtokens,tagset='universal')
        for (token,pos) in tagged:
            if pos in ['NOUN','VERB','ADJ','ADV','NUM']:
                tokens = listadd(tokens, (lemma(token,pos_replace[pos],use_pos),pos_replace[pos]) if lemmatize else (token,pos_replace[pos]))
    return tokens

def hypernym_reduce(tokens,use_pos=False):
    reduced = list(tokens)
    pointer = len(reduced)
    while pointer > 0:
        t = reduced[-pointer]
        for u in reduced:
            if u != t:
                if hypernym(t[0],u[0],word1_pos=t[1],word2_pos=t[1],use_pos=use_pos):
                    reduced.remove(t)
                    break
        pointer -= 1
    return reduced

def hypernym(word1,word2,word1_pos=None,word2_pos=None,use_pos=False):
    if word2 == None:
        return True
    if word1 == None:
        return False
    if word1 == word2 and (not use_pos or word1_pos == word2_pos):
        return True
    synsets1 = wn.synsets(word1,pos=word1_pos) if ((word1_pos != None) and use_pos) else wn.synsets(word1)
    synsets2 = wn.synsets(word2,pos=word2_pos) if ((word1_pos != None) and use_pos) else wn.synsets(word2)
    for s1 in synsets1:
        for s2 in synsets2:
            if hypernym_search(s2,s1):
                return True
    return False

def hypernym_search(synset,target):
    for h in synset.hypernyms():
        if h == target or hypernym_search(h,target):
            return True
    return False

def breadcrumb_screening(text1,text2,relax=0,use_pos=False):
    set1 = hypernym_reduce(tokenset(text1),use_pos=use_pos)
    set2 = hypernym_reduce(tokenset(text2),use_pos=use_pos)
    union = hypernym_reduce(listunion(set1,set2),use_pos=use_pos)
    return ((len(listdiff(union,set1))) <= relax, (len(listdiff(union,set2))) <= relax)

def keyword_string(text,lemmatize=True,use_pos=False,shuffle=False):
    ts = list(hypernym_reduce(tokenset(text,lemmatize=lemmatize,use_pos=use_pos)))
    if shuffle:
        shuffle(ts)
    string = ts[0][0]
    if len(ts) > 1:
        for token in ts[1:]:
            string += ', '
            string += token[0]
    return string

def hyper(word1,word2):
    if word1 == None:
        return word2
    if word2 == None:
        return word1
    if hypernym(word1[0],word2[0]):
        return word1
    elif hypernym(word2[0],word1[0]):
        return word2
    else:
        return None

def hypo(word1,word2):
    if word1 == None or word2 == None:
        return None
    if hypernym(word1[0],word2[0]):
        return word2
    elif hypernym(word2[0],word1[0]):
        return word1
    else:
        return None

def common_parent(text1,text2):
    set1 = tokenset(text1)
    set2 = tokenset(text2)
    intersection = set()
    for token1 in set1:
        addition = None
        for token2 in set2:
            if hypernym(token1[0],token2[0]):
                addition = hyper(addition,token1)
            if hypernym(token2[0],token1[0]):
                addition = hyper(addition,token2)
        if addition != None:
            intersection = listadd(intersection, addition)
    return intersection
                

def to_string(tset):
    if tset == set():
        return 'None'
    tset = list(tset)
    string = tset[0][0]
    if len(tset) > 1:
        for token in tset[1:]:
            string += ', '
            string += token[0]
    return string

def tokenset_neg_check(set1,set2,relax=2):
    union = hypernym_reduce(listunion(set1,set2))
    return (len(listdiff(union,set1))) >= min(relax,len(set2)) and (len(listdiff(union,set2))) >= min(relax,len(set1))