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
#data inputs
#im = io.imread(fullpath)

x = np.arange(0, 350)
x, y = comp.fake_pmt_n(x)
fig, ax = plotting.plot_pmt_nonlinearity(x, y)



fake_ratio = 2.
fake_true_photons, fake_green_channel = comp.fake_pmt_n(np.arange(0,140,fake_ratio))

fake_x2, fake_red_channel = comp.fake_pmt_n(fake_true_photons/fake_ratio, round=False)
channel_i = 0
channel_j = 1
fig, ax, title = plotting.plot_channels(fake_green_channel, fake_red_channel, channel_i, channel_j)

```


```python id="82a5927b"
#data manipulation

xs, ys, xs_per_y = comp.get_unmixing_ratio(fake_green_channel, fake_red_channel)
fig, ax, title = plotting.plot_unmixing_vectors(xs, ys, channel_i, channel_j, label='FakeFP', plot=True)

detected_photons, true_photons = comp.compute_PMT_nonlinearity(fake_green_channel, fake_red_channel, xs_per_y)
#whats the best way to handle X possibly being larger than y? we want to get the same curve either way. 

#print(fake_green_channel)
#print(detected_photons)

#print(true_photons)
fig, ax = plotting.plot_pmt_nonlinearity(true_photons, detected_photons)


#fullpath = os.path.join(current_data_dir, file)
#for i in range(im.shape[3]):
#  for j in range(im.shape[3]):
```


```python id="f700a7f6"
#plots
new_y = list(range(6,149))
new_x = []
for i in new_y:
    new_x.append(comp.correct_PMT_nonlinearity(i, y, x))

fig, ax = plotting.plot_pmt_nonlinearity(new_x, new_y)


```


```python id="8dd23ba7"
#data output

```
