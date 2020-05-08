# DCGAN and SAGAN
pytorch DCGAN with wasserstein distance and pytorch SAGAN (spectral normalisation and attention on conv GAN)

This repo gives an implementation of DCGAN and SAGAN with wasserstein distance in pytorch . The dataset used is CIFAR 10 some of the images are as follows :

For DCGAN

![Generated_Images](https://github.com/rsn870/DCGAN/blob/master/images/DCGAN/fake_samples.JPG?raw=true)

Graph for losses for DCGAN  is as follows :

![Wasserstein_Losses ](https://github.com/rsn870/DCGAN/blob/master/images/DCGAN/wasserstein.JPG?raw=true)

FID plot for DCGAN is as follows :

![FID for DCGAN](https://github.com/rsn870/DCGAN/blob/master/images/DCGAN/FID.JPG?raw=true)

FID has been computed for every 100 epochs when total training was set for close to 100000 epochs

For SAGAN

![Generated_Images](https://github.com/rsn870/DCGAN/blob/master/images/SAGAN/fake_samples.png?raw=true)

FID plot for SAGAN ias as follows :


![FID for SAGAN](https://github.com/rsn870/DCGAN/blob/master/images/SAGAN/FID.JPG?raw=true)

FID has been computed for every 100 epochs when total training was set for close to 100000 epochs





