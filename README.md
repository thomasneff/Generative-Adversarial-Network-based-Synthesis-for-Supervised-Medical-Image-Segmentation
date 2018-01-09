# Generative-Adversarial-Network-based-Synthesis-for-Supervised-Medical-Image-Segmentation
Code for the paper 'Generative Adversarial Network based Synthesis for Supervised Medical Image Segmentation'

This modification adds the ability to generate pixel-wise segmentations to the GAN. Currently, it assumes that the images are grayscale, therefore the GAN model only handles 2 image channels (1 for the image, 1 for the segmentation)

For more details, check out our [paper](http://castor.tugraz.at/doku/OAGM-ARWWorkshop2017/oagm-arw-17_paper_30.pdf).

For citations ([bibtex](cite.bib)):
```
Neff, Thomas and Payer, Christian and Stern, Darko and Urschler, Martin (2017). 
Generative Adversarial Network based Synthesis for Supervised Medical Image Segmentation.
In Proceedings of the OAGM&ARW Joint Workshop, pp. 140-145.
```

# Credits
Credits to [sugyan](https://github.com/sugyan) for their [tf-dcgan](https://github.com/sugyan/tf-dcgan) implementation, which this code is based on.

# How to use/modify
In train.py, the 'idlist_\*' variables defined at the start of 'train' need to point to text files containing a list of image/segmentation filenames, one for each line.

The '\*_folder_path\*  variables need to point to the folder containing the files defined in the idlists.

(You can easily change the data loading, the dcgan model just takes image batches as a tensor as input.)


