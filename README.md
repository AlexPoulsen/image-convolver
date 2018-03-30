# image-convolver
version 2.4 of my image convolver. previous versions were not on github unfortunately.

running requires numpy, scipy, math, os, matplotlib, and inspect

run time on multichannel images (i.e. color vs grayscale) shouldn't be much more than run time on single channel images.

single channel tests for image/feature pairings similar to the ones in the below image (21x21px image and 3x3px feature) have an average run time of ~37.5ms. more time tests are to come.

## examples
for small images, the progress bar is partially disabled so that every convolution has a print() that says it is running. otherwise they may be too fast for all of the progress bars to print, even with print set to flush output

![](o_x_demostration.png)