

## Triplet Network in Pytorch

The following repository contains code for training Siamese and Triplet Network in Pytorch
Siamese and Triplet networks make use of a similarity metric with the aim of bring similar images closer in the embedding space while separating non similar ones.
Popular uses of such networks being - 
* Face Verification / Classification
* Learning deep embeddings for other tasks like classification / detection / segmentation

### Installation
---
``` 
pip install -r requirements.txt
```

### Training
---
``` 
python train.py --cuda
```
This by default will train on the MNIST dataset

### MNIST / FashionMNIST
---
``` 
python train.py --cuda --dataset fmnist
```
Parameter **exp_name** to be passed as either **mnist** or **fmist**

### [VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/)
---
 Specify the location of the dataset in test.yaml
 ```
 python train.py --exp_name VGGFace2_exp1 --cuda --epochs 50 --ckp_freq 5 --dataset vggface2 --num_train_samples 32000 --num_test_samples 5000 --train_log_step 50
 ```

### [Street2Shop](http://www.tamaraberg.com/street2shop/)
---
 Under testing ...

### TODO
---
- [x] Train on MNIST / FashionMNIST
- [x] Train on a public dataset
- [x] Multi GPU Training
- [x] - [ ] Include popular models - ResneXT / Resnet / VGG / Inception