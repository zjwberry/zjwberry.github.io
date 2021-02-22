'''
LastEditors: jingweizhu
'''
import requests
import random
import math
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear, Embedding, Conv2D
import numpy as np

TEXT8_LOCAL_PATH = "./data/text8.txt"
SAMPLE_LOCAL_PATH = "./data/sample.txt"

def download():
    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request = requests.get(corpus_url)
    corpus = web_request.content
    with open(TEXT8_LOCAL_PATH, "wb") as f:
        f.write(corpus)
    f.close()

def load_text8():
    with open(TEXT8_LOCAL_PATH, "r") as f:
        corpus = f.read().strip()
    f.close()
    return corpus

def data_preprocess(corpus):
    corpus = corpus.strip().lower()
    corpus = corpus.split(" ")
    return corpus

def build_dict(corpus):
    word_freq_dict = {}
    for word in corpus:
        if word not in word_freq_dict:
            word_freq_dict[word] = 0
        word_freq_dict[word] += 1
    
    word_freq_dict = sorted(word_freq_dict.items(), key = lambda x:x[1], reverse=True)
    word2id_dict = {}
    word2id_freq = {}
    id2word_dict = {}

    for word, freq in word_freq_dict:
        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[curr_id] = freq
        id2word_dict[curr_id] = word
    
    return word2id_freq, word2id_dict, id2word_dict

def convert_corpus_to_id(corpus, word2id_dict):
    corpus = [word2id_dict[word] for word in corpus]
    return corpus

def subsampling(corpus, word2id_freq):
    def discard(word_id):
        return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / word2id_freq[word_id] * len(corpus))
    corpus = [word for word in corpus if not discard(word)]
    return corpus

def build_data(corpus, word2id_dict, word2id_freq, output_fp, max_window_size = 3, negative_sample_num = 4):
    count = 0
    corpus_size = len(corpus)

    for center_word_idx in range(len(corpus)):
        window_size = random.randint(1, max_window_size)
        center_word = corpus[center_word_idx]

        positive_word_range = (max(0, center_word_idx - window_size), min(corpus_size - 1, center_word_idx + window_size))
        positive_word_candidates = [corpus[idx] for idx in range(positive_word_range[0], positive_word_range[1] + 1) if idx != center_word_idx]

        for positive_word in positive_word_candidates:
            output_fp.write("%d\t%d\t1\n" % (center_word, positive_word))
            count += 1

            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)
                if negative_word_candidate not in positive_word_candidates:
                    output_fp.write("%d\t%d\t0\n" % (center_word, negative_word_candidate))
                    count += 1
                    i += 1
    return count

""" ---- ** ----
# step1:
#download()

# step2:
corpus = load_text8()

# step3:
corpus = data_preprocess(corpus)

# step4:
word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_dict)
print "there are %d words in the corpus" % (vocab_size)
for word, id in word2id_dict.items()[:50]:
    freq = word2id_freq[id]
    print "word [%s], ID [%d], freq [%d]" % (word, id, freq)

# step5:
corpus = convert_corpus_to_id(corpus, word2id_dict)
print "%d tokens in the corpus at all " % len(corpus)
print corpus[:50]

# step5:
corpus = subsampling(corpus, word2id_freq)
print "%d tokens in the corpus after subsampling" % len(corpus)
print corpus[:50]

# step6:
fp = open(SAMPLE_LOCAL_PATH, "w")
count = build_data(corpus, word2id_dict, word2id_freq, fp)
fp.close()
---- ** ---- """


with dygraph.guard():
    USR_ID_NUM = 10
    usr_emb = Embedding(size=[USR_ID_NUM, 16], is_sparse=False)
    arr = np.random.randint(0, 10, (3)).reshape((-1)).astype("int64")
    print arr
    arr_pd = dygraph.to_variable(arr)
    emb_res = usr_emb(arr_pd)
    print emb_res.numpy()
    print emb_res.shape