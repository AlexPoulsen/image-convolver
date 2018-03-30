# image-convolver
version 2.4 of my image convolver. previous versions were not on github unfortunately.

running requires numpy, scipy, math, os, matplotlib, and inspect

run time on multichannel images (i.e. color vs grayscale) shouldn't be much more than run time on single channel images.

times averaged over 1000 runs

* 39.086 ms 21x21 3x3 gs
* 73.444 ms 21x21 3x3 mono -> clr
* 103.0 ms 21x21 3x3 mono -> clr -> mono
* 178.695 ms 21x21 5x5 mono x clr
* 252.252 ms 21x21 5x5 mono x clr -> mono
* 459.035 ms 21x21 9x9 mono x clr
* 641.333 ms 21x21 9x9 mono x clr -> mono
* 673.694 ms 21x21 3x3 clr x mono
* 702.674 ms 21x21 3x3 clr x mono -> mono
* 776.033 ms 21x21 5x5 clr
* 845.151 ms 21x21 5x5 clr -> mono
* 1035.856 ms 21x21 9x9 clr
* 1227.949 ms 21x21 9x9 clr -> mono
* 0.009847820609725371 ms, approximate time per pixel

## examples
for small images, the progress bar is partially disabled so that every convolution has a print() that says it is running. otherwise they may be too fast for all of the progress bars to print, even with print set to flush output

![](o_x_demostration.png)