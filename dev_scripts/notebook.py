# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/GreggHeller1/Neuron_Tutorial/blob/main/scripts/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + id="71ee021b"
#settings
# %load_ext autoreload
# %autoreload 2
try:
  import google.colab
  in_colab = True
except:
  in_colab = False
print(in_colab)

# + colab={"base_uri": "https://localhost:8080/"} id="4e02e926" outputId="84475a29-508b-4d96-adf5-e85665e994d2"
#installs (for colab only, run this once)
if in_colab:
    # ! git clone https://github.com/GreggHeller1/PMT_linearization.git


# + id="5e9731ca"
#local imports
#cwd if in colab for imports to work
if in_colab:
    # %cd /content/PMT_linearization

from src import data_io as io
from src import plotting
from src import computation as comp
from src import main


# + id="db51ef2e"
#imports
from matplotlib import pyplot as plt
import os
import numpy as np

# + colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="989c69e2-c8c4-43e0-9ba6-7a36f66be4c3"
#define paths
#cwd if in colab for file loading to work
if in_colab:
    # %cd /content/PMT_linearization/scripts
    
test_path = os.path.join('demo_data', 'test.txt')
print(test_path)
print(os.getcwd())
os.path.exists(test_path)

# + colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="56f159c6-3dbc-4b37-d217-083fb5d2e792"

fp = 'FakeFP'
x = np.arange(0, 350)
x, y = comp.fake_pmt_n(x)
fig, ax = plotting.plot_pmt_nonlinearity(x, y)



fake_ratio = 2.
fake_true_photons, fake_green_channel = comp.fake_pmt_n(np.arange(0,140,fake_ratio))

fake_x2, fake_red_channel = comp.fake_pmt_n(fake_true_photons/fake_ratio, round=False)
channel_i = 0
channel_j = 1
fig, ax, title = plotting.plot_channels(fake_green_channel, fake_red_channel, channel_i, channel_j, alpha=1, label=fp)
io.savefig(fig, title)


# + id="82a5927b"
#data manipulation

xs, ys, xs_per_y = comp.get_unmixing_ratio(fake_green_channel, fake_red_channel)
fig, ax, title = plotting.plot_unmixing_vectors(xs, ys, channel_i, channel_j, label=fp, plot=True)
io.savefig(fig, title)
print(xs_per_y)
print(xs, ys)


# +

detected_photons, true_photons = comp.compute_PMT_nonlinearity(fake_green_channel, fake_red_channel, xs_per_y)
io.save_PMT_curve(detected_photons, true_photons, i=channel_i, j=channel_j,fp = fp)



#Plot the inferred nonlinearity and see if it matches
fig, ax = plotting.plot_pmt_nonlinearity(true_photons, detected_photons)
io.savefig(fig, f'PMT curve from {fp} on {channel_i}{channel_j}')


# +

#use the inferred nonlinearity to correct both channels then plot. The result should be linear
corrected_green = []
corrected_red = []

for g,r in zip(fake_green_channel, fake_red_channel):
    try:
        corrected_green.append(comp.correct_PMT_nonlinearity(g, detected_photons, true_photons))
        corrected_red.append(comp.correct_PMT_nonlinearity(r, detected_photons, true_photons))
    except Exception as E:
        print(g,r)

fig, ax, title = plotting.plot_channels(corrected_green, corrected_red, channel_i, channel_j, alpha=1, label=f'{fp}_corrected')
io.savefig(fig, title)


# + id="f700a7f6"
#Now lets do it with an image of actual flourophores

filename = "TDTom_only.tif"#"TFP_10us_dwell_915nm_Z_STACK.tiff"#"YFP_10us__STACK.tiff"#
filepath = os.path.join('demo_data', filename)
os.path.exists(filepath)


im = io.imread(filepath)
fp = comp.fp_from_tiffname(filename)

for i in range(im.shape[3]):
    for j in range(im.shape[3]):
        if not(i == j):
            try:
                main.main(fp, i, j, im[:,:,:,i].flatten(), im[:,:,:,j].flatten(), alpha=.01)
            except Exception as E:
                pass



# + id="8dd23ba7"
curve_dict = io.load_PMT_curves() 
fig, ax = plotting.plot_PMT_curves(curve_dict)
io.savefig(fig, f'PMT_nonlinearities from {fp}')
# -
#use 0,1 YFP, 12, YFP, 01 TFP Only
l = ('a', 'b')
os.path.join(*l)



main.main('FakeFP', 4, 6, fake_red_channel, fake_green_channel, alpha=1)

# +
a = np.array((1,2,3,4))
b = np.array((1,3,6,7))

x = 2.3
idx = np.searchsorted(a, x)-1




# -

y = np.array([1,2,3,4,5,6,7,8,9,10])
print(comp.smooth(y,1))

funclist = [lambda x:x, lambda x:x**2]
x = np.array([1,2,3,4,5])
np.piecewise(x, [x<3, x>=3], funclist)

np.searchsorted(x,y)-1

y[np.logical_and(y>3, y<7)]


np.isnan(1)

# +
path = "/Users/Gregg/Dropbox (MIT)/Files for Gregg/FromJoe/4color_testing/unmixingcoeffs/tdTomato_invivo_run2_10usdwell_915nm_700mWbeforeobject/rawtif/run2_10usdwell_915nm_700mWbeforeobject__STACK.tiff"
im = io.imread(path)


# -

cropped = im[1,:,:,:]
cropped.shape

plt.imshow(cropped[:,:,2])

from mpl_toolkits.mplot3d import Axes3D

# +

valid_mask = cropped[:,:,2]>4
cropped_nonzero = cropped[valid_mask,:]
a = np.arange(0,cropped_nonzero.shape[0],1)
subsample = np.random.choice(a, 10000)
subsampled =  cropped_nonzero[subsample,:]
print(subsampled.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cropped_nonzero[:,0], cropped_nonzero[:,1], cropped_nonzero[:,2])
ax.view_init(elev=10., azim=0)



# +

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cropped_nonzero[:,0], cropped_nonzero[:,1], cropped_nonzero[:,2])
ax.view_init(elev=30., azim=-20)
# -

mask = cropped[:,:,2]<4*cropped[:,:,0]
cropped[mask]=0

plt.imshow(cropped[:,:,2])

# +
valid_mask = cropped[:,:,2]>4
cropped_nonzero = cropped[valid_mask,:]
a = np.arange(0,cropped_nonzero.shape[0],1)
subsample = np.random.choice(a, 10000)
subsampled =  cropped_nonzero[subsample,:]
print(subsampled.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(cropped_nonzero[:,0], cropped_nonzero[:,1], cropped_nonzero[:,2])
ax.view_init(elev=10., azim=0)
# -

plotting.plot_channels(cropped[:,:,0], cropped[:,:,1], 2,1, alpha=.002)


