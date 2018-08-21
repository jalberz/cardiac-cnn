# clinical-informatics-project

[Kaggle Second National Data Science Bowl](https://www.kaggle.com/c/second-annual-data-science-bowl)

Declining cardiac function is a key indicator of heart disease. Doctors determine cardiac function by measuring end-systolic and end-diastolic volumes (i.e., the size of one chamber of the heart at the beginning and middle of each heartbeat), which are then used to derive the ejection fraction (EF). EF is the percentage of blood ejected from the left ventricle with each heartbeat. Both the volumes and the ejection fraction are predictive of heart disease. While a number of technologies can measure volumes or EF, Magnetic Resonance Imaging (MRI) is considered the gold standard test to accurately assess the heart's squeezing ability.

## Folder Layout

```
.
├── cardiac_cnn
│   ├── log
│   │   ├── log_128linear.txt
│   │   ├── log_128.txt
│   │   ├── log_2.txt
│   │   ├── log_3pool.txt
│   │   ├── log_3.txt
│   │   ├── log_4.txt
│   │   ├── log_alexnet.txt
│   │   ├── log_deep.txt
│   │   ├── log_extralog.txt
│   │   ├── log_googlenet.txt
│   │   ├── log_linear.txt
│   │   ├── log_mlpnet.txt
│   │   ├── log_mlp.txt
│   │   ├── log_nodropout.txt
│   │   ├── log_softrelu.txt
│   │   ├── log_tanh.txt
│   │   ├── log.txt
│   │   └── log_updated.txt
│   ├── Predicting EF through Image Processing.pdf
│   ├── src
│   │   ├── preprocessing.py
│   │   ├── train_128linear.py
│   │   ├── train_128.py
│   │   ├── train_3pool.py
│   │   ├── train_6pool.py
│   │   ├── train_alexnet.py
│   │   ├── train_deep.py
│   │   ├── train_googlenet.py
│   │   ├── train_linear.py
│   │   ├── train_mlp.py
│   │   ├── train.py
│   │   ├── train.R
│   │   ├── train_softrelu.py
│   │   └── train_updated.py
│   ├── submission
│   │   ├── submission_alexnet.csv
│   │   ├── submission_googlenet.csv
│   │   ├── submission_lenet.csv
│   │   └── submission_mlp.csv
│   ├── submission_alexnet.csv
│   ├── submission_lenet.csv
│   └── submission_mlp.csv
├── LICENSE
└── README.md

4 directories, 42 files
```

## Neural Network Types Implemented
- [MLP](http://deeplearning.net/tutorial/mlp.html)
- [LeNet](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)
- [AlexNet](http://vision.stanford.edu/teaching/cs231b_spring1415/slides/alexnet_tugce_kyunghee.pdf)
- [GoogLeNet](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/googlenet.html)


## Links

- [Mxnet python machine learning library](https://mxnet.apache.org/)
- [Mxnet tutorials on convolutional neural network models](https://github.com/apache/incubator-mxnet/tree/master/example/image-classification)
- [Winning Kaggle solution breakdown](https://datasciencebowl.com/leading-and-winning-team-submissions-analysis/)
- [Understanding Activation Functions in Neural Networks](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0)
- [Analysis of Kaggle solutions](https://github.com/jonrmulholland/dsbAnalysis)
 


Final group project for applied clinical informatics
