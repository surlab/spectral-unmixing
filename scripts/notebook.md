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
<a href="https://colab.research.google.com/github/GreggHeller1/Neuron_Tutorial/blob/main/scripts/notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="71ee021b"
#settings
%load_ext autoreload
%autoreload 2
try:
  import google.colab
  in_colab = True
except:
  in_colab = False
print(in_colab)
```

```python colab={"base_uri": "https://localhost:8080/"} id="4e02e926" outputId="84475a29-508b-4d96-adf5-e85665e994d2"
#installs (for colab only, run this once)
if in_colab:
    ! git clone https://github.com/GreggHeller1/PMT_linearization.git
```


```python id="5e9731ca"
#local imports
#cwd if in colab for imports to work
if in_colab:
    %cd /content/PMT_linearization

from src import data_io as io
from src import plotting
from src import computation as comp
from src import main
```


```python id="db51ef2e"
#imports
from matplotlib import pyplot as plt
import os
import numpy as np
```

```python colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="989c69e2-c8c4-43e0-9ba6-7a36f66be4c3"
#define paths
#cwd if in colab for file loading to work
if in_colab:
    %cd /content/PMT_linearization/scripts
    
test_path = os.path.join('demo_data', 'test.txt')
print(test_path)
print(os.getcwd())
os.path.exists(test_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="56f159c6-3dbc-4b37-d217-083fb5d2e792"

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
```


```python id="82a5927b"
#data manipulation

xs, ys, xs_per_y = comp.get_unmixing_ratio(fake_green_channel, fake_red_channel)
fig, ax, title = plotting.plot_unmixing_vectors(xs, ys, channel_i, channel_j, label=fp, plot=True)
io.savefig(fig, title)
```


```python

detected_photons, true_photons = comp.compute_PMT_nonlinearity(fake_green_channel, fake_red_channel, xs_per_y)
#whats the best way to handle X possibly being larger than y? we want to get the same curve either way. 

#Plot the inferred nonlinearity and see if it matches
fig, ax = plotting.plot_pmt_nonlinearity(true_photons, detected_photons)
io.savefig(fig, f'PMT curve from {fp} on {channel_i}{channel_j}')
```


```python

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
```


```python id="f700a7f6"
#Now lets do it with an image of actual flourophores

filename = "YFP_10us__STACK.tiff"
filepath = os.path.join('demo_data', filename)
os.path.exists(filepath)

im = io.imread(filepath)
fp = comp.fp_from_tiffname(filename)

for i in range(im.shape[3]):
    for j in range(im.shape[3]):
        if j>i:
            main.main(fp, i, j, im[:,:,:,i].flatten(), im[:,:,:,j].flatten(), alpha=.01)


```


```python id="8dd23ba7"
#data output

filename = "YFP_10us__STACK.tiff"
idx = filename.lower().find('fp')
filename[idx-1:idx+2]
```
```python

l = ('a', 'b')
os.path.join(*l)

```


```python
main.main('FakeFP', 4, 6, fake_red_channel, fake_green_channel, alpha=1)
```

```python

```
