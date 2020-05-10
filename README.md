# DCGAN and SAGAN
pytorch DCGAN with wasserstein distance and pytorch SAGAN (spectral normalisation and attention on conv GAN)

This repo gives an implementation of DCGAN and SAGAN with wasserstein distance in pytorch . The dataset used is CIFAR 10 some of the images are as follows :

For DCGAN

![Generated_Images](https://github.com/rsn870/DCGAN/blob/master/DCGAN/fake_samples.JPG?raw=true)

Graph for losses for DCGAN  is as follows :

![Wasserstein_Losses ](https://github.com/rsn870/DCGAN/blob/master/DCGAN/wasserstein.JPG?raw=true)

FID plot for DCGAN is as follows :

![FID for DCGAN](https://github.com/rsn870/DCGAN/blob/master/DCGAN/FID.JPG?raw=true)

FID has been computed for every 100 epochs when total training was set for close to 100000 epochs

Evolution for fake images sampled every 10 epcohs for around 300 or so epochs is present in zip file in DCGAN directory in images 

Weights for some epochs are present in models /DCGAN folder

For SAGAN

![Generated Images](https://github.com/rsn870/DCGAN/blob/master/SAGAN/fake_samples.png?raw=true)

FID plot for SAGAN ias as follows :

![FID for SAGAN](https://github.com/rsn870/DCGAN/blob/master/SAGAN/FID.JPG?raw=true)



FID has been computed for every 40 epochs when total training was set for close to 100000 epochs

Visualisation of the attention map is also presented :

![Attention Map](https://github.com/rsn870/DCGAN/blob/master/SAGAN/att_map.JPG?raw=true)

Some saliency map for couple of images choosen in the dataset (original images were appropriately resized for ease of view)

![Saliency Map](https://github.com/rsn870/DCGAN/blob/master/SAGAN/saliency_map.JPG?raw=true)





