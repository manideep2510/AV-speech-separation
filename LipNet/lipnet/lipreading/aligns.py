import numpy as np

class Align(object):
    def __init__(self, absolute_max_string_len=128, label_func=None):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(float((y[1]))*25, float((y[2]))*25, y[0]) for y in [x.strip().split(" ") for x in lines[4:]]]
        self.build(align)
        return self

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        self.align = align
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[0]/25 <4.8] )

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)


class Align_1(object):
    
    def __init__(self, absolute_max_string_len=128, label_func=None, length=5, index=2):
        self.label_func = label_func
        self.absolute_max_string_len = absolute_max_string_len
        self.length=length
        self.index=index

    def from_file(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        align = [(float((y[1])), float((y[2])), y[0]) for y in [x.strip().split(" ") for x in lines[4:]]]
        #print(self.index)
        #print(align[self.index])
        p = None
        for i in align[self.index:]:
            p = i
            if((i[1]-align[self.index][0]) >= self.length): break
        
        end_index=align.index(p)

        #print(self.index)
        
        self.start=int(align[self.index][0]*25)
        self.end=int(align[end_index-1][1]*25)
        
        align=align[self.index:end_index]
        self.build(align)
        
        return self
    
    def video_range(self):
        return [self.start,self.end]

    def from_array(self, align):
        self.build(align)
        return self

    def build(self, align):
        self.align = align
        self.sentence = self.get_sentence(align)
        self.label = self.get_label(self.sentence)
        if(len(self.label)>self.absolute_max_string_len):self.label=self.label[:self.absolute_max_string_len]
        self.padded_label = self.get_padded_label(self.label)

    def strip(self, align, items):
        return [sub for sub in align if sub[2] not in items]

    def get_sentence(self, align):
        return " ".join([y[-1] for y in align if y[1]] )

    def get_label(self, sentence):
        return self.label_func(sentence)

    def get_padded_label(self, label):
        padding = np.ones((self.absolute_max_string_len-len(label))) * -1
        return np.concatenate((np.array(label), padding), axis=0)

    @property
    def word_length(self):
        return len(self.sentence.split(" "))

    @property
    def sentence_length(self):
        return len(self.sentence)

    @property
    def label_length(self):
        return len(self.label)
