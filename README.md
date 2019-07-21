

## Triplet Network in Pytorch

The following repository contains code for training Siamese and Triplet Network in Pytorch
Siamese and Triplet networks make use of a similarity metric with the aim of bring similar images closer in the embedding space while separating non similar ones.
Popular uses of such networks being - 
* Face Verification / Classification
* Learning deep embeddings for other tasks like classification / detection / segmentation

### Installation
---
Install [PyTorch](https://pytorch.org/get-started/locally/)
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
python train.py --cuda --dataset <manist>/<fmnist>
```
To create a **tSNE** visualisation
``` 
python tsne.py --ckp <path to model>
```
The embeddings and the labels are stored in the experiment folder as a pickle file, and you do not have to run the model everytime you create a visualisation. Just pass the saved embeddings as the --pkl parameter
``` 
python tsne.py --pkl <path to stored embeddings>
```
Sample tSNE visualisation on MNIST 
![tSNE](images/tSNE_mnist.jpg "tSNE visualisation on MNIST")
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
- [ ] Include popular models - ResneXT / Resnet / VGG / Inception