#!/usr/bin/env python3

import sys
import os
import numpy as np
import scipy 
import matplotlib.pyplot as plt
from sklearn import decomposition
from PIL import Image
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
ap.add_argument("-c", "--components", type = int,default=3,
	help = "No Of Components to be Used for Compression")


args = vars(ap.parse_args())

img=args['image']
dims=(600,600)
k=args['components']

im =  np.array(Image.open(img).resize(dims))
#im =  np.array(Image.open(img).convert('L').resize(dims))
im=im/255.0
flatim=im.flatten()
flatim=im.flatten().reshape(-1,3)
u, s, v = decomposition.randomized_svd(flatim, k)
low_rank = u @ np.diag(s) @ v
low_rank=low_rank.reshape(dims+(3,))
data_needed=k*dims[0]+dims[1]*k+k#size of v,d +k
print("\t\tUSING {} COMPONENTs and {} DATA POINTS OUT OF ORIGNAL {} POINTS.".format(k,data_needed,dims[0]*dims[1]))

low_rank=(low_rank*255.0).astype(np.uint8)
low_rank=Image.fromarray(low_rank)
low_rank.save("compressed.png")
