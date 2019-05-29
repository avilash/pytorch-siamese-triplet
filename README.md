
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

### [Street2Shop](http://www.tamaraberg.com/street2shop/)
---
 Under testing ...

### TODO
---
- [x] Train on MNIST / FashionMNIST
- [ ] Train on [Street2Shop](http://www.tamaraberg.com/street2shop/) dataset
- [ ] Include popular models - ResneXT / Resnet / VGG / Inception
- [ ] Multi GPU Training