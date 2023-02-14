---
jupyter:
  jupytext:
    formats: ipynb,py,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region colab_type="text" id="view-in-github" -->
<a href="https://colab.research.google.com/github/GreggHeller1/PMT_linearization/blob/main/scripts/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="71ee021b" tags=[]
#settings
%load_ext autoreload
%autoreload 2
try:
  import google.colab
  in_colab = True
except:
  in_colab = False
print(f'Session is in colab: {in_colab}')
```

```python colab={"base_uri": "https://localhost:8080/"} id="4e02e926" outputId="84475a29-508b-4d96-adf5-e85665e994d2" tags=[]
#installs (for colab only, run this once)
if in_colab:
    ! git clone https://github.com/GreggHeller1/PMT_linearization.git
```


```python id="5e9731ca" tags=[]
#local imports
#cwd if in colab for imports to work
if in_colab:
    %cd /content/PMT_linearization
    
from src import data_io as io
from src import plotting as plot
from src import computation as comp
from src import main
from src import config as gcfg
```


```python id="db51ef2e" tags=[]
#imports

from matplotlib import pyplot as plt
import os
import numpy as np
#import xarray as xr
#import pandas as pd
```


```python colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="989c69e2-c8c4-43e0-9ba6-7a36f66be4c3" tags=[] jupyter={"source_hidden": true}
#define paths
#cwd if in colab for file loading to work
if in_colab:
    %cd /content/PMT_linearization/scripts
    
#test_path = os.path.join('demo_data', 'test.txt')
#print(test_path)
#print(os.getcwd())
#os.path.exists(test_path)

```


```python
#Unmixing parameters - these override the defaults in app_config.py
#if you are not changing parameters frequently you should set them in app_config.py and delete them here
cfg = main.UnmixingSession()

#bead image
#cfg.open_path = "/Users/Gregg/Dropbox (MIT)/For Gregg/Gregg_Kendyll_unmixing/4C/4C_Run1_10us"
#I16 for sample 3 color
cfg.open_path = "/Users/Gregg/Dropbox (MIT)/For Gregg/020223/Newunmixing_testing/3C_PVsynTD_TG_YFP"
#for windows use r"Pasted/path/here"

supdir, filename = os.path.split(cfg.open_path)
cfg.save_path = cfg.open_path if os.path.isdir(cfg.open_path) else supdir
cfg.filename = os.path.splitext(filename)[0]
#savepath = "/Users/Gregg/Dropbox (MIT)/For Gregg/Gregg_Kendyll_unmixing/4C/4C_Run1_10us"

cfg.linearize_PMTs = False                 #True or False

cfg.past_correctible_range = 'max'        #zero, max, correct
#zero: set the pixel value to zero in all channels
#max: set the pixel values in all channels to the maximum corrected value. This should make it obvious which pixels to ignore
#correct: use the best fit curve to attempt to linearize them anyway. This should never make the estimates WORSE

cfg.unmix= True                           #True or False
#Either load the coefficients from files
#flourophores = ['BFB', 'YGFB', 'RFB', 'DRFB']
#unmixing_coefficients = io.get_unmixing_mat(flourophore_list = flourophores)

#Or enter them manually
#FOR SAMPLE CELL
unmixing_coefficient_dict = {
    'TFP': [.622325, .34619, .031483],
    'YFP': [.199198, .707837, .092694],
    'RFP': [0.06666, .092964, .77324],
    }

#FOR BEADS
#unmixing_coefficient_dict = {
#    'BFB': [.620, .331, .035, .013],
#    'YGFB': [.179, .741, .066, .013],
#    'RFB': [0.0, .003, .799, .197],
#    'DRFB': [0.0, 0.0, 0.0, 1.0],
#    }
unmixing_mat = []
for fp, coefs in unmixing_coefficient_dict.items():
    unmixing_mat.append(coefs)
cfg.unmixing_mat= np.array(unmixing_mat).T
cfg.num_channels = cfg.unmixing_mat.shape[0]

cfg.handle_negatives =  'set_to_zero'     #'set_to_zero', 'non_negative_least_squares' or None
#set_to_zero: run the unmixing as normal and then set all negative pixel values to 0
#non_negative_least_squares: use a different algorithm that allows only positive values (https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html)
#None: Return negative values

cfg.save_original_tiff = False          #True or False

cfg.save_processed_tiff = True         #True or False

cfg.compression = None               #None or LZW, maybe others? unclear documentation...
```

```python colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="56f159c6-3dbc-4b37-d217-083fb5d2e792"
#data inputs
#io.readfile(test_path)
image, tif_tags = io.imread(cfg.open_path, num_channels=cfg.num_channels, verbose=True)
```


```python id="82a5927b"
#data manipulation/
print(f"The image is {image.shape[3]} channels,{image.shape[1]}x{image.shape[2]} pixels and {image.shape[0]} Z frames")
new_image, residuals = main.process_image(cfg, image)
```


```python id="8dd23ba7"
#data output
io.umixing_app_save(cfg, image, new_image)
```


```python id="f700a7f6"
#plots
#a single channel of original and processed images
channel = 2
frame = 50

plot.paired_images_single_channel(image, new_image, frame, channel)

```


```python
#plots
#plot all channels of a single frame of the unmixed image
plot.single_frame_all_channels(image, frame, channel)

#Probably want to do some sort of normalization on these so the same intensity is the same color. 
#What color scale would be useful?

```


```python
#load images to compare matlab and python
matlab_unmixed = "/Users/Gregg/Dropbox (MIT)/For Gregg/020223/Newunmixing_testing/3C_PVsynTD_TG_YFP/Unmixed/1045_8-17-2022_cell1_close__STACK_unmixed.tiff"
matlab_original = "/Users/Gregg/Dropbox (MIT)/For Gregg/020223/Newunmixing_testing/3C_PVsynTD_TG_YFP/Nounmixing_justtiffs/1045_8-17-2022_cell1_close__STACK.tiff"

python_original = "/Users/Gregg/Dropbox (MIT)/For Gregg/020223/Newunmixing_testing/3C_PVsynTD_TG_YFP/3C_PVsynTD_TG_YFP_original.tiff"
python_unmixed = "/Users/Gregg/Dropbox (MIT)/For Gregg/020223/Newunmixing_testing/3C_PVsynTD_TG_YFP/3C_PVsynTD_TG_YFP_unmixed.tiff"


matlab_original_im, tif_tags = io.imread(matlab_original, num_channels=cfg.num_channels, verbose=True)
matlab_unmixed_im, tif_tags = io.imread(matlab_unmixed, num_channels=cfg.num_channels, verbose=True)


python_original_im, tif_tags = io.imread(python_original, num_channels=cfg.num_channels, verbose=True)
python_unmixed_im, tif_tags = io.imread(python_unmixed, num_channels=cfg.num_channels, verbose=True)

```


```python
#compare matlab and python images
im1 = matlab_unmixed_im
im2 = python_unmixed_im
new_image = np.subtract(im1.astype(np.int16), im2.astype(np.int16))
print(new_image.dtype)
sum_diff = np.sum(np.absolute(new_image))
sum_pixels1 = np.sum(np.absolute(im1))
sum_pixels2 = np.sum(np.absolute(im2))

print(f'sum_diff: {sum_diff}')
print(f'sum_pixels: {sum_pixels1}')
print(f'sum_pixels: {sum_pixels2}')
print(f'fraction err: {sum_diff/sum_pixels}')


a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(a-b)

print(np.amax(im1.astype(np.int16)))
print(np.sum(im1==164))

print(np.amax(im2.astype(np.int16)))
print(np.sum(im2==197))



print(np.amin(new_image.astype(np.int16)))
print(new_image.shape)
```


```python
#plots for the comparison
#a single channel of original and processed images
channel = 1
frame = 190

image=matlab_unmixed_im#[1:,:,:,:]
new_image = python_unmixed_im#[:-1,:,:,:]

plot.paired_images_single_channel(image, new_image, frame, channel)

```


```python jupyter={"outputs_hidden": true} tags=[]
#use for observing pairs of pixel values to see if the algorithm seems to be working

#V This can be a bit slow, comment out the first 4 lines if necessary
indecies = np.unravel_index(new_image.argmin(), new_image.shape)
print(indecies)
print(image[:,indecies[1], indecies[2], indecies[3]])
print(new_image[:,indecies[1], indecies[2], indecies[3]])

print('######################')

frame = 1
chan_x_end = 20
chan_y = 3

for chan_x in range(chan_x_end):
    pixel = image[frame,chan_x,chan_y,:]
    print(f'pixel: {pixel}')
    unmixed_pixel = new_image[frame,chan_x,chan_y,:]
    print(f'unmixed_pixel: {unmixed_pixel}')
    print(f'unmixed dot prod: {np.dot(unmixing_mat, unmixed_pixel)}')
    x_inferred, res, rank, s = np.linalg.lstsq(unmixing_mat, pixel)
    print(f'inferred directly: {x_inferred}')
    print(f'direct dot prod: {np.dot(unmixing_mat, x_inferred)}')
    print('######################')

```


```python tags=[] jupyter={"source_hidden": true, "outputs_hidden": true}
#code was used to help generate the algorithms with a dummy image

tup = (1,2,3,4)

im = np.array([[[[1,0],[0,1,],[1,0,]],
               [[0,1],[1,1],[0,1]],
               [[1,0],[0,1],[1,0]]],#####
               [[[1,0],[0,1,],[1,0,]],
               [[0,1],[2,2],[0,1]],
               [[1,0],[0,1],[1,0]]],#####
               [[[1,0],[0,1,],[1,0,]],
               [[0,1],[3,3],[0,1]],
               [[1,0],[0,1],[1,0]]],#####
               [[[1,0],[0,1,],[1,0,]],
               [[0,1],[6,6],[0,1]],
               [[1,0],[0,1],[1,0]]]]#####
             )
#we have made an image with 2 channels - 3x3 pixels and 4 z/t frames 
#+ shape in the first channel and x shape in the second channel
#when we reshape this we want to end up with 2x36, and be able to put the pluses and xs back
print(im.shape)               
total_pixels = np.prod(np.array(im.shape[:-1]))
print(f'Total pixels: {total_pixels}')

reorder_axis = np.moveaxis(im, -1, 0)
new_im= np.reshape(reorder_axis, (reorder_axis.shape[0], total_pixels))
print(reorder_axis.shape)
print(new_im)
old_im = np.reshape(new_im, reorder_axis.shape)
old_im = np.moveaxis(old_im, 0, -1)
print(old_im.shape)   
print((old_im==im).all())
#print(old_im)
#print(im)

channel = 1
frame = 3
cropped = old_im[frame,:,:,:]
cropped.shape

plt.imshow(cropped[:,:,channel])

```
