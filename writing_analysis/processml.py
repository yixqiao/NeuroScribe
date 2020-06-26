import warnings, pickle
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import *
from tensorflow import Graph, Session
from tensorflow.keras.backend import set_session

import string
import math
import re

tf.config.threading.set_intra_op_parallelism_threads(16)
tf.config.threading.set_inter_op_parallelism_threads(16)
graph = tf.get_default_graph()
sess = tf.Session()



# Tokenizer
d1, d2 = pickle.load(open('writing_analysis/tokenizer_410.dat', 'rb'))
def tokenW(n):
    try:
        return d1[n]
    except:
        return ''
def tokenN(s):
    try:
        return d2[s]
    except:
        return 0
def tokenWL(nums):
    words = ""
    for i in range(len(nums)):
        words += tokenW(nums[i]) + " "
    return words
def tokenNL(words):
    ws = words.split()
    ar = np.empty((len(ws),))
    for i in range(len(ws)):
        ar[i] = tokenN(ws[i])
    return ar
        
def cleanup(post):
    goodChars = string.ascii_letters + string.digits + string.punctuation
    punc = """ .,'"!() """
    stPunc = string.punctuation
    
    post = post.strip()
    post = post.translate(str.maketrans("""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""", " " * len("""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~"""))) # No apostrophes                

    ps = post.split()
    ps2 = ps # Split punctuation

    ps3 = [] # Remove only punctuation
    for p in ps2:
        allPunc = True
        for c in p:
            if not c in stPunc:
                allPunc = False
                break

        if allPunc:
            continue
        ps3 += p.split()

    return " ".join(ps3)

def load_resources():
  # define model
  with graph.as_default():
    set_session(sess)
    model = load_model('writing_analysis/model_410.hdf5')
    # compile network
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model._make_predict_function()
  return model

model = load_resources()

stopwords = """a about above after again against all am an and any are aren't as at be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split()
stopwords += "urllink de di la en".split()

def processpost(text):
    max_length = 10
    ml2 = max_length//2
    
    text = text
    text_lower = " " + text.lower() + " "
    
    encoded_full = tokenNL(cleanup(text_lower))
    
    encoded_text = tokenWL(encoded_full)
    
    encoded = []
    out = []
    
    prevLoc = 0

    inputdata = np.zeros((len(encoded_full),10))
    for i in range(len(encoded_full)):
        inputdata[i, max(ml2-i,0):min(len(encoded_full)-i+ml2,max_length)] = encoded_full[max(i-ml2,0):min(i+ml2,len(encoded_full))]
        inputdata[i, 5] = 0

    # predict probabilities for each word
    with graph.as_default():
        set_session(sess)
        yhat_all = model.predict(inputdata, batch_size=32, verbose=0)
        
    for i in range(len(encoded_full)):
        yhat_all
        yhat = yhat_all[i]
                    
        curword = tokenW(encoded_full[i])
        
        evalv = yhat[int(encoded_full[i])]
        bestv = np.amax(yhat)

        maxw = np.argmax(yhat)
        while(maxw==0 or tokenW(maxw).strip() in stopwords):
            yhat[maxw] = -1000
            maxw = np.argmax(yhat)

        maxw = tokenW(maxw)
        
        #print("%-8s - %8.4f (%6.3f / %6.3f) %s" % (curword, yhat[0,int(encoded_full[i])]/np.amax(yhat) * 100,
        #                                      yhat[0,int(encoded_full[i])] * 100, np.amax(yhat) * 100, maxw))
        

        # print("%6.3f / %6.3f" % (yhat[0,int(encoded_full[i])] * 100, np.amax(yhat) * 100))
        
        prevEnd = prevLoc
        searchResult = re.search("[^a-z]" + curword + "[^a-z]", text_lower[prevLoc:])
        if curword != "" and searchResult is not None:
          prevEnd = searchResult.start() + prevLoc
        
        if(prevLoc < prevEnd):
            out.append({"word": text[prevLoc:prevEnd], "eval": -1, "best": -1, "bestw": "", "alpha": any(i in text[prevLoc:prevEnd] for i in string.ascii_letters)})
        out.append({"word": text[prevEnd:prevEnd+len(curword)], "eval": evalv, "best": math.pow(bestv, 2), "bestw": maxw})
        if searchResult is not None:
          prevLoc = prevEnd + len(curword)
    out.append({"word": text[prevLoc:], "eval": -1, "best": -1, "bestw": "", "alpha": any(i in text[prevLoc:] for i in string.ascii_letters)})
    return out
