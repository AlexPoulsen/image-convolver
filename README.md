# image-convolver
version 3.0 of my image convolver.

running requires numpy, math, os, matplotlib, inspect, time, typing, atexit, and imageio, including standard lib packages.

run time on multichannel images (i.e. color vs grayscale) shouldn't be much more than run time on single channel images.

updated time testing will be added soon.

## examples
for small images, the progress bar is partially disabled so that every convolution has a print() that says it is running. otherwise they may be too fast for all of the progress bars to print, even with print set to flush output. The numbers that set whether it prints normally, adds a delay on each print to let it flush, or removing the loading part, are dependant on the hardware you run it on. I will be working on a function that will generate them for you based on your hardware, but for the time being, you should change them yourself to suit your computer.

For an image demonstration, please see the test images directory until i get around to making another demo pic. The replaced part of the algorithm starting in v3 has changed the output around edges, which is now more correct, but it is slightly different than before, so the old demo image is not usable any more.