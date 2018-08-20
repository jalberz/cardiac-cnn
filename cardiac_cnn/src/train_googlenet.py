# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Training script, this is converted from a ipython notebook
"""

import os
import csv
import sys
import numpy as np
import mxnet as mx
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# In[2]:

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def get_googlenet(num_classes = 600, **kwargs):
    data = mx.sym.Variable("data")
    conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv1")
    pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
    conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1,1), name="conv2")
    conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1,1), name="conv3")
    pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

    in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
    in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
    pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
    in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
    in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
    in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
    in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
    pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
    in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
    pool6 = mx.sym.Pooling(in5b, kernel=(7, 7), stride=(1,1), global_pool=True, pool_type="avg")
    flatten = mx.sym.Flatten(data=pool6)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
    return mx.symbol.LogisticRegressionOutput(data=fc1, name='softmax')

def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


# In[3]:

def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    systole = label_data[:, 1]
    diastole = label_data[:, 2]
    systole_encode = np.array([
            (x < np.arange(600)) for x in systole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return systole_encode, diastole_encode

def encode_csv(label_csv, systole_csv, diastole_csv):
    systole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    np.savetxt(systole_csv, systole_encode, delimiter=",", fmt="%g")
    np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("./train-label.csv", "./train-systole.csv", "./train-diastole.csv")


# # Training the systole net

# In[4]:

network = get_googlenet()
batch_size = 32
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="./local_train-64x64-data.csv", data_shape=(30, 64, 64),
                           label_csv="./train-systole.csv", label_shape=(600,),
                           batch_size=batch_size)

data_test = mx.io.CSVIter(data_csv="./local_test-64x64-data.csv", data_shape=(30, 64, 64), label_csv="./train-systole.csv", label_shape=(600,), batch_size=batch_size) 

data_validate = mx.io.CSVIter(data_csv="./validate-64x64-data.csv", data_shape=(30, 64, 64),
                              batch_size=1)

systole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 65,
        learning_rate      = 0.01,
        wd                 = 0.001,
        momentum           = 0.9)

# systole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

# TODO: ADD EVAL TEST TO DIASTOLE
systole_model.fit(X=data_train, eval_data=(data_test), eval_metric = mx.metric.np(CRPS))



# # Predict systole

# In[5]:

systole_prob = systole_model.predict(data_validate)


# # Training the diastole net

# In[6]:

network = get_googlenet()
batch_size = 32
devs = [mx.gpu(0)]
data_train = mx.io.CSVIter(data_csv="./local_train-64x64-data.csv", data_shape=(30, 64, 64),
                           label_csv="./train-diastole.csv", label_shape=(600,),
                           batch_size=batch_size)

data_test = mx.io.CSVIter(data_csv="./local_test-64x64-data.csv", data_shape=(30, 64, 64), label_csv="./train-diastole.csv", label_shape=(600,), batch_size=batch_size) 

diastole_model = mx.model.FeedForward(ctx=devs,
        symbol             = network,
        num_epoch          = 65,
        learning_rate      = 0.01,
        wd                 = 0.001,
        momentum           = 0.9)

# diastole_model.fit(X=data_train, eval_metric = mx.metric.np(CRPS))

# TODO: ADD EVAL TEST TO DIASTOLE
diastole_model.fit(X=data_train, eval_data=(data_test), eval_metric = mx.metric.np(CRPS))



# # Predict diastole

# In[7]:

diastole_prob = diastole_model.predict(data_validate)


# # Generate Submission

# In[8]:

def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(1570):
        line = fi.__next__() # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result


# In[9]:

systole_result = accumulate_result("./validate-label.csv", systole_prob)
diastole_result = accumulate_result("./validate-label.csv", diastole_prob)


# In[10]:

# we have 2 person missing due to frame selection, use udibr's hist result instead
def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h
train_csv = np.genfromtxt("./train-label.csv", delimiter=',')
hSystole = doHist(train_csv[:, 1])
hDiastole = doHist(train_csv[:, 2])


# In[11]:

def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p



# In[12]:

fi = csv.reader(open("data/sample_submission_validate.csv"))
f = open("submission_googlenet.csv", "w")
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.__next__()) # Python2: fo.writerow(fi.next())
for line in fi:
    idx = line[0]
    key, target = idx.split('_')
    key = int(key)
    out = [idx]
    if key in systole_result:
        if target == 'Diastole':
            out.extend(list(submission_helper(diastole_result[key])))
        else:
            out.extend(list(submission_helper(systole_result[key])))
    else:
        print("Miss: %s" % idx)
        if target == 'Diastole':
            out.extend(hDiastole)
        else:
            out.extend(hSystole)
    fo.writerow(out)
f.close()

