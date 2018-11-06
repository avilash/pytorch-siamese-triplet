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
python train.py
```

### TODO
---
- [ ] Support for a public dataset
- [ ] Include popular models - ResneXT / Resnet / VGG / Inception
- [ ] Multi GPU Training

If you need the dataset used in training please write to me at avilashkumar4[at]gmail[dot]com