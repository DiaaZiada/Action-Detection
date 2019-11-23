
# Action Detection 
Action Recognition is ipython notebook contains all steps of training action recognition model with optical flow and single person tracking 

****Requirements****
 - [Python](https://www.python.org/) 3.*
 - [Imutils](https://pypi.org/project/imutils/)
 - [Numpy](http://www.numpy.org/)
 - [OpenCV](https://opencv.org/)
 - [Pytorch](https://pytorch.org/)
 - [Caffe2](https://caffe2.ai)
 - [Scikit-learn](https://scikit-learn.org/stable/)
 - [SciPy](https://www.scipy.org)
 - [Matplotlib](https://matplotlib.org)
   
## Dataset
* [MERL_Shopping_Dataset](ftp://ftp.merl.com/pub/tmarks/MERL_Shopping_Dataset)
# Work 
the implementation of the model was based on on [A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection](http://www.merl.com/publications/docs/TR2016-080.pdf) paper

the model consist of three stages 

 1. person detection and tracking 
 2. optical flow 
 3. action detection
 
 ## Person Detection & Tracking 
for this part we peruse [pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd) repo.
we faced some problems cause of the view angle of the camera, some time model didn't notice the person as person or mistake the dimensions of him  but we solved by those steps:
 1. take highest object's score as a person, and ignoring the detection label of it
 2. make fixed box instead of dynamic one
 3. for the missing person frames, keep the previous one as the current 

## Optical Flow 
for this part we peruse [flowiz](https://github.com/georgegach/flowiz) repo.
the problem w faced at this point was the output of the repo wasn't good enough when we use two consecutive frames so that we decided to take `frame[n]` and `frame[n-6]` to calculate the optical flow in frame n
## Projection layer
![projection layer](https://github.com/DiaaZiada/action-detection/blob/master/image/Screenshot%20from%202019-11-23%2012-34-00.png)

## Action Recognition
for this part we peruse [Action Recognition](https://github.com/eriklindernoren/Action-Recognition) repo.
model consist of:

 - vgg16 net as features extractor (encoder)
 - lstm net with attention mechanism (decoder)
 
 trained by make prediction every 6 frames
 ![projection layer](https://github.com/DiaaZiada/action-detection/blob/master/image/Screenshot%20from%202019-11-23%2012-34-22.png)


## Models
models in this project are based on [Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper
model architecture: 

![mini exception cnn model](https://github.com/DiaaZiada/Faces/blob/master/images/mini_exception_cnn_model.png)

project consist of 3 models:
	

 1. Gender
 2. Expression
 3. Multiple

### Gender Model
trained done by using [IMDB](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) , [WIKI](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/),  [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/)  datasets. consist of ~ 600 K image, and by using tencrop data augmentation dataset increased to be ~ 6 M image 
Accuracy ~ 78% using only 6 epochs and it will reach higher accuracy expected to be ~ 96 %

### Expression Model
trained done by using [FER](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) , [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/), [JAFFE](http://www.kasrl.org/jaffe.html)  image datasets. consist of ~ 40 K image, and by using tencrop data augmentation dataset increased to be ~ 400 K image 
Accuracy ~ 60% using 15 epochs and it will reach higher accuracy expected to be ~ 66 %

### Multiple Models
this model is little bit different form Gender & Expression models 
in this model we use one feature extractor model and 5 different classifiers each classifier predict specific features form faces and they are:
* Illumination : _Bad, Medium, High_
*  Pose : _Frontal, Left, Right, Up, Down_
*  Occlusion : _Glasses, Beard, Ornaments, Hair, Hand, None, Others_
*  Age : _Child, Young, Middle and Old_
*  Makeup : _Partial makeup, Over-makeup_

trained done by using [IMFDB](http://cvit.iiit.ac.in/projects/IMFDB/) image datasets consist of ~ 3 K image, and by using tencrop data augmentation dataset increased to be ~ 30 K image 
Accuracy ~ 77% using 15 epochs
## Train
all training process done on [Faces notebook](https://github.com/DiaaZiada/Faces/blob/master/Faces.ipynb) using [Google Colab](https://colab.research.google.com) cloud 
## Credits

[Real-time Convolutional Neural Networks for Emotion and Gender Classification](https://arxiv.org/pdf/1710.07557.pdf) paper

[Simple object tracking with OpenCV by  pyimagesearch](https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/)

