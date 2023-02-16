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

<!-- #region id="view-in-github" colab_type="text" -->
<a href="https://colab.research.google.com/github/GreggHeller1/PMT_linearization/blob/main/scripts/demo_notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<!-- #endregion -->

```python id="71ee021b" outputId="9b3f40fa-fde9-4d6c-cbef-4420600351c8" colab={"base_uri": "https://localhost:8080/"}
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

```python id="4e02e926" colab={"base_uri": "https://localhost:8080/"} outputId="44daa2e8-6edc-4548-d196-2483cb2bbd46"
#installs (for colab only, run this once)
if in_colab:
    ! git clone https://github.com/GreggHeller1/PMT_linearization.git

```

```python id="5e9731ca" colab={"base_uri": "https://localhost:8080/"} outputId="219a09ac-17e7-41f7-c241-e44349447ff6"
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
#import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import os
import numpy as np

```

```python colab={"base_uri": "https://localhost:8080/"} id="a06b6e4a" outputId="d28207c9-abb8-43c0-a0e8-7bef7a44e754"
#define paths
#cwd if in colab for file loading to work
if in_colab:
    %cd /content/PMT_linearization/scripts
    
test_path = os.path.join('demo_data', 'test.txt')
print(test_path)
print(os.getcwd())
os.path.exists(test_path)
```

```python colab={"base_uri": "https://localhost:8080/"} id="b3586a50" outputId="86b36ec8-cdb6-4830-c773-a083d4784577"
#data inputs
io.readfile(test_path)



```

<!-- #region id="kQfTUlaSeE_J" -->
#Spectral unmixing for multiple flourophores in-vivo
When doing in-vivo 2 photon imaging, resolving multiple flourophores is often assumed to require spectrally distinct flourophores with a seperatne detection channel for each. i.e. To image a red flourophore (tdtomato) and a green flourophore (GFP) that emit photons at two distinct wavelengths (580 and 508). Photons can be seperated by wavelength using dichroic mirrors and filters so that those photons above 550 (mostly from tdtomato) go to one detector, and those below 550 (mostly from GFP) go to a different detector. 

* But what if the emission spectra are mostly overlapping? Even with well seperated flourophres there is often some overlap (e.g. some photons emitted by GFP will be detected in the red channel) 

* or what if you want to distinguish more colors with a limited number of detectors/PMTs? 

Spectral unmixing can accomplish this. 
<!-- #endregion -->

```python id="p4_SR2EGrvIq" outputId="35183ec6-4b38-4421-d8fe-c22ebe2f76a0" colab={"base_uri": "https://localhost:8080/", "height": 834}
#load in excitation and emission for all 4
def load_exc_em(csv_name):
  path = os.path.join('demo_data', csv_name)
  df = pd.read_csv(path)
  return df
#create the red and green filters
#total range is 300:949
#green is 470:550
#red can be 550:650

spectra_mapping = {
    'YFP': 'YFP.csv',
    'CFP': 'CFP.csv',
    'GFP': 'GFP.csv',
    'mOrange': 'mOrangeFP.csv',
    'mCherry': 'mCherryFP.csv',
    'TdTomato': 'tdTomatoFP.csv',
    'Blue FluoSpheres': 'BFS.csv',
    'Crimson FluoSpheres': 'CrFS.csv',
    'Dark Red FluoSpheres': 'DRFS.csv',
    'Nile Red FluoSpheres': 'NRFS.csv',
    'YellowGreen FluoSpheres': 'YGFS.csv',
    'Orange FluoSpheres': 'OFS.csv',
  }

def load_all_spectra(spectra_mapping):
  spectra_dict = {}
  for FP_name, csv_name in spectra_mapping.items():
    spectra_dict[FP_name] = load_exc_em(csv_name)
  return spectra_dict


spectra_dict = load_all_spectra(spectra_mapping)

def create_filter(wavelength, bandwidth, max_em=100, steepness=10000):
  #steepness is the input to a sigmoid that is used to symettrically shape either end of the filter
  #steepness must be greater than 1
  assert(steepness>1)
  range = max(bandwidth, 50)
  wavelengths = np.arange(wavelength-range, wavelength+range, 1)
  up_sigmoid = 1/(1+steepness**(-(wavelengths-(wavelength-bandwidth/2))))
  down_sigmoid = 1-(1/(1+steepness**(-(wavelengths-(wavelength+bandwidth/2)))))
  emission = np.minimum(up_sigmoid, down_sigmoid)
  emission = emission*max_em
  df = pd.DataFrame({'Wavelength':wavelengths, 'Emission':emission})
  return df

filter_dict = {}
color_dict = {}
filter_dict['Red Filter'] = create_filter(600, 100, max_em = 100, steepness=10)
color_dict['Red Filter'] = 'r'
filter_dict['Green Filter'] = create_filter(500, 100, max_em = 100, steepness=10)
color_dict['Green Filter'] = 'g'


#then we plot the excitation and emission curves, and the filters
def plot_spectrum(ax, wavelength, values, label='', color=None, hatch='', alpha=.2):
  if color is None:
    ax.plot(wavelength, values, label=label)
    #Fill under the curve
    ax.fill_between(
            x= wavelength, 
            y1= values, 
            alpha= alpha, hatch=hatch)
  else:
    ax.plot(wavelength, values, label=label, color=color)
    #Fill under the curve
    ax.fill_between(
            x= wavelength, 
            y1= values, 
            color= color,
            alpha= alpha, hatch=hatch)
  return ax


FP_list = ['GFP', 'TdTomato']
Filter_list = ['Green Filter', 'Red Filter'] 

plot_ex_em_spectra(FP_list, Filter_set_list=Filter_list)

FP_list = ['GFP', 'CFP']
plot_ex_em_spectra(FP_list, Filter_set_list=Filter_list)

FP_list = ['GFP', 'mOrange', 'mCherry']
plot_ex_em_spectra(FP_list, Filter_set_list=Filter_list)


```

<!-- #region id="WzinugxAg62q" -->
#How does spectral unmixing work?

Spectral unmixing relies on the property that the distribution of photon wavelengths emitted by a flourophore is consistent and independant of the total number of photons emmitted by that type of flourophore. If you know that all the photons collected must've been produced by specific flourophores, then the photons collected at a given excitation pixel must be a linear combination of the photons that can be produced by individual flourophores. As long as the number of effective detection channels meets or exceeds the number of flourophores, and that the effective spectra of the flourophores are LINEARLY INDEPENDANT then we can use linear algebra to solve for the amounts of the flourophores. 

We solve the linear system of equations 
$$
\textbf{A} \vec{x} = \vec{b}
$$
For $C$ detection channels, $\vec{b}$ is the vector of length $C$ with the $c$th entry representing the photons detected in the $c$th channel for a given pixel. For $N$ flourophores, $\textbf{A}$ is an $C$x$N$ matrix where each entry $A_{c,n}$ is the fraction of total emmision from the $c$th flourophore that will be detected in $n$th channel. $\vec{x}$ is a vector of length $N$ with the $n$th entry being the inferred amount of the $n$th flourophore. 


<!-- #endregion -->

<!-- #region id="uxs--BkV_3f8" -->
![unmixing_graphic.jpg](data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAlgCWAAD/4QB0RXhpZgAATU0AKgAAAAgABAEaAAUAAAABAAAAPgEbAAUAAAABAAAARgEoAAMAAAABAAIAAIdpAAQAAAABAAAATgAAAAAAAACWAAAAAQAAAJYAAAABAAKgAgAEAAAAAQAABdygAwAEAAAAAQAAA0sAAAAA/+0AOFBob3Rvc2hvcCAzLjAAOEJJTQQEAAAAAAAAOEJJTQQlAAAAAAAQ1B2M2Y8AsgTpgAmY7PhCfv/iAkBJQ0NfUFJPRklMRQABAQAAAjBBREJFAhAAAG1udHJSR0IgWFlaIAfQAAgACwATADMAO2Fjc3BBUFBMAAAAAG5vbmUAAAAAAAAAAAAAAAAAAAAAAAD21gABAAAAANMtQURCRQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACmNwcnQAAAD8AAAAMmRlc2MAAAEwAAAAa3d0cHQAAAGcAAAAFGJrcHQAAAGwAAAAFHJUUkMAAAHEAAAADmdUUkMAAAHUAAAADmJUUkMAAAHkAAAADnJYWVoAAAH0AAAAFGdYWVoAAAIIAAAAFGJYWVoAAAIcAAAAFHRleHQAAAAAQ29weXJpZ2h0IDIwMDAgQWRvYmUgU3lzdGVtcyBJbmNvcnBvcmF0ZWQAAABkZXNjAAAAAAAAABFBZG9iZSBSR0IgKDE5OTgpAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABYWVogAAAAAAAA81EAAQAAAAEWzFhZWiAAAAAAAAAAAAAAAAAAAAAAY3VydgAAAAAAAAABAjMAAGN1cnYAAAAAAAAAAQIzAABjdXJ2AAAAAAAAAAECMwAAWFlaIAAAAAAAAJwYAABPpQAABPxYWVogAAAAAAAANI0AAKAsAAAPlVhZWiAAAAAAAAAmMQAAEC8AAL6c/8AAEQgDSwXcAwEiAAIRAQMRAf/EAB8AAAEFAQEBAQEBAAAAAAAAAAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHwJDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqSk5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/EAB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEHYXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdoaWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj5OXm5+jp6vLz9PX29/j5+v/bAEMAAgICAgICAwICAwQDAwMEBQQEBAQFBwUFBQUFBwgHBwcHBwcICAgICAgICAoKCgoKCgsLCwsLDQ0NDQ0NDQ0NDf/bAEMBAgICAwMDBgMDBg0JBwkNDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDQ0NDf/dAAQAXv/aAAwDAQACEQMRAD8A/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAor8YPC/xC/bw/aA/aI+OfgH4R/FfRvCeifDHxGun29vqWhWV0TbXc12kKJILSR28tbQ7i7EnI5PNdX48+If/AAUZ/ZJ0aT4qfFO/8KfGXwHp7Idcg0u0/szVLC2LAG4TyreBNq5G5iswUEllVQXAB+ulFcR8NfiH4Y+LPgHQfiT4MuDdaL4jsYb+zkYBXCSjJSRQTtkjbKSLk7XUjtXSnWtHXUhozX1sNQZPMFoZk88p/e8vO7HvigDSorwz9pX4vz/AX4G+L/izZWMOqXnh3Tzc21lcSmGOeVnSNQzKC21S+4gDJAxkZyOh+CHji++JPwc8C+PtXNsNT8R+GtH1a/jtMiGO6v7OKeVUUs7KodyFDMSB1J60AepUVmnWtHXUhozX1sNQZPMFoZk88p/e8vO7HvitKgAorI/4SDQRqn9hnUrT+0sbvsfnx/aMYznyt2/GOelfK37cH7SGt/su/Ay6+IvhbTrLVNZm1C00y0hvnYQRNdl/37pGQ8oQJ9wMmSR8wxggH2BRUUM8NwnmW8iypnG5GDDP1FUrfWtHu76fS7W+tpry2AM1vHMjTRA9N6All/ECgDSoor5n/ai+Kuo+BP2cfiT4++Gus2qa/wCGNInuLe4h8i8+y3Sbcb43EiZGfuup+lAH0xRXi/7OPi/X/iB+z/8ADjx14ruRea14g8K6PqeoXAjSIS3V3aRyyuEjVUTc7E7VUKOgAFe0UAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH/9D9/KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA8x8UfG34M+B9Xfw/wCNPH3hjQNUiRJHstU1mzsrlUkGUYxTSo4DDkEjBHSud/4ab/Zt/wCiseB//Cj07/5Ir8x/Enwh+Gvxr/4KveMfCPxU0C28R6PD8O7W+jtLveI1uYjYoknyMpyFkYdcc19vf8O9/wBjD/olOi/ncf8Ax6gD6fvPHfgfTtQ0XSNQ8Q6VbX3iQM2jWs17BHPqQRVZjaRs4a4AVlJ8sNgEHuK6uvys/ay0+z0n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VkZ/v3dqMKJj33A4Vh271+ogIPINfyneEtRvv2Vv+CjsukaY5g06PxLJp8iA7UktNQkCgEdCNzAj0xX9ViNkIw6EZ/PGPyzQBNRRRQAUUUUAeFftNru+AHjsf9Qec/lg1+bn/BN4f8Xf8aH/AKl6w/8ASmev0r/aSjMvwE8eKO2h3jf98oT/AEr81/8Agm8P+Lt+Mm9dAsR+VzNVLYD9laKKKkAooooAKKKKACiiigAooooAKKKKACiiigAooooAK/Hn4u/P/wAFBdGX+7rvhn9IrU1+w1fjz8Tv3/8AwUN0+Mf8s9d8O/8AjtpaPTQH7DUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPzH/4Kyf8mpah/wBhC3/9Cr3f9gz/AJNT8A/9g8fzrwj/AIKyf8mpah/2ELf/ANCr3f8AYM/5NT8A/wDYPH86APsGiiigAooooAKKKKACiiigD8C/24v+Ui/wl+mnf+lBr99K/Av9uL/lIv8ACX6ad/6UGv30oAKKKKACiiigAooooAKKKKACiiigAooooA/AT9rr/lJr8MP+3L/0M1+/dfgJ+11/yk1+GH/bl/6Ga/fugAooooAK84+Jvwr8E/F/wzL4P8facupaXM4d4WZlyy9MMpUivR6KAPhT/h3F+yV/0Jif+BEv+NH/AA7i/ZK/6ExP/AiX/GvuuigD4U/4dxfslf8AQmJ/4ES/40f8O4v2Sv8AoTE/8CJf8a+66KAPFvg/8Avhf8CdOutM+GWjrpEF9IJLgK7SF2X13M36V7TRRQAUUUUAFfgPa/8AKYN/rH/6TGv34r8B7X/lMG/1j/8ASY0AfvxRRRQAUUUUAFFFFABRRRQAUUUUAFMl/wBU/wDun+VPpkv+qf8A3T/KgD8Df2BP+UgXxj/66ah/6PSv31r8Cv2BP+UgXxj/AOumof8Ao9K/fWgAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/0P38ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKguRcG2lFoVWco3llxlQ+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KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA/9H9/KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK/Nj/gox4EN94V8NfEW1jzJpV1Jpl2yjkwXY3xM3+ykkZUe8lfpPXlnxt8Br8TPhT4m8FBA8+oWEn2TPa7hxLbn2HnIufamgMH9m7x5/wALG+CfhXxJLJ5l2LJbK9JOW+02RMEjN6Fym/6MK9xr8t/+Cc/j1oz4p+Fl+5R0ZdZs424II2290MHpg+Scf7xr9SKGAUUUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiszVdb0bQrY3muX9rp1uOst3MkCcf7TkD9a8c1L9pf4Iafcmxt/FFvq93/Db6LFNqsjH0As45h+ZAoA4f4j+N/C/w6/aK8PeJPGd4dN0yXwdqdmly0MsqG4e+tHCfukc5Kox6dvpXkP7T/wAVvhT8Xvhl/wAIj4T8W2K339o211m7t72KPy4g4b5ltXOfmGBivof/AIXn4g1b5PBvwv8AGOpufuSX9tBo1uw7HfezRvj/ALZ0f29+05rfzad4U8KeGU9NX1a41GUf8As4I0J9vM/GssTh4V6UqNTZqxE4KcXF9T8ZP+FZWv8A0OPh789Q/wDkGj/hWVr/ANDj4e/PUP8A5Br9m/8AhBv2hNX+fWPibYaOrfeg0Lw/EcDvtmvZrg/+OZo/4UBJffN4k+I3jrVS33411cWEDe3l2UUBx7FjXzv+p+X/AN77/wDgHH/ZtHzPyG8Gfs/+KfiDr93oPhHVtHvjY2Ivri7MtxBaohk8sRl5rdD5mecbcbec9QK3xA+AutfDyCyuPEPifwsRcX0VqyWmp/aZod4YmWSKOMyCJdvzMFOCQMc1+w8P7LnwMEiz6h4b/taccmXVb271BifU/aZ5B+QxXe6T8IvhVoODo3g3QbJl6PBptsj/AIsI9xP1NXS4VwFOanG9/UccvpJ3R+A7+EvDyHH/AAnGgOfSOPVH/wDQdPNY11odtCcWmq2196GCC9UH/v7bR1/SfBBBbRiG2jSKNeiIoVR9AOKlrP8A1Qy/+99//AF/ZtHzP5mk0rUpDhLc/VmRP/QmFdN4b+Gvivxn4g0rw1oEEU+o6lOYoYfOTgqjOS7AlVUBTyTX9H9FaUuFcDTlzRvfXr3Vuw45fRTuj8NPE37GPxr8N+FdW8T6tBpyWulWc15PHHd+bMY4ULsEVEbc2BwMj615LB8M7aSCN28X6BGWRSUb+0NykjocWJGR3wTX9EtFH+quB9mqetr33/rsH9n0bWP54P8AhWFr/wBDl4e/PUf/AJBrs/CHwg+HWoNdr42+Jmm6MECfZWsdOvtQMjHdvDhorbYF4xgtnPbHP72UUUuE8vhLm5W/VhHLqKd7H5AeJr3wv4d+BOpfDvwl8Y7jWoFtGSDQLfw8lvHdyzTCR0897dp/mdiwJlzwBnHFffv/AAqH4lSf8fHxf8SN/wBc7LS4/wCVoa+gqK+hp0404KnBWS0R2xioqyPy3+FfiLwb4d8GQ6H4n+M3ijwvqNpf6qk+lWttH5cDfb7ghsnT5TmVSJW+cjc5xgcD0T/hOPhQfv8A7QnjT8I0X/3FV+glFaXGfn1/wmvwi7ftB+N8+4Q/+4ql/wCE5+FQ+5+0J4y/GGM/+4qv0EoouB8L/s9fCnxlqfwZ8LanY/EfxNocdxaO6WEEdi0EIM0mAgmtWfDfe+Zj19K85+3S/Dj4vfELSvFPxi1nw9dyzaQ6Xkmk2t2+pL9gj+dwtnJGvk58sbAuQOcnJr9LqKLgfn1/wt2zj/1H7R07enn+Ebd//QLWOj/hddxFzD+0Fpk2P4bjwVLz9THIn8q/QWii4H59f8NF+IbT/j3+LHgy/wAf8/XhjVoc/XypjTvgbr/x18S6p498UeBpPBmq22o6/G9xcXS6nYwTzJZW6brUFJXWMKFBEgLbwe2K/QOii4H51/E/4pfFX4Z/F3wp4v8AiD4d0APHourWdutlql0LaYTS2rPulaxdkddg2qUIYE5Ixzox/t42ULhb/wAFSsvdrHVYZx+Amityfyr9A6KLgfDtp+3r8Lnwuo+HvEto3crb2s6D8Uui3/jtdfYfts/s+3Qze6vf6afS60u7J/8AIMUor6zoougPhbSf2h/gp4h/aNtvE2n+LbCLSo/BdzpzXd95mnxC7fUIJRFm6SL5jGC3pgGuz+P3xG8BXfh3wdreneIdN1Cy0zxtoF9dy2FzHeeVbQTlpJCsBkbao64Br6l1DSNJ1aPydVsre9j6bbiJJV/JgRXm2rfAX4Ka5uOpeBvD8jt1kTToIpD/AMDjRW/WgDl/+GqPgH/0Ncf/AIB3n/xij/hqj4B/9DXH/wCAd5/8Ypn/AAy/8JbT5/DcGr+G5ez6NrWoWe36ItwYx+C0f8KR8W6Z8/hX4r+MLRx91dTks9XiH4XNtvI+rmjQB/8Aw1R8A/8Aoa4//AO8/wDjFeZePvjH8N/iZ4p+GOh+BtYGq31r43sb2aGO2uIylvHaXiPITJEi4VpFB5zz6Zr0r/hHf2mtF403xj4X8SL6axo09g59t9ncMuffy/wo/wCE3/aG0X5Nb+G2na2q/en8P69GvHfbDfQwMfpvzQB9C0V88/8ADROlaZ/yOfgzxl4aRf8AWT3eiy3Vqvr++sTcqR+R9q6jw98fvgt4ocQ6P4z0dpycC3uLlbS4z6eTceXJn220WA9eoqOGaG4iWe3dZY3GVdCGVh6gjg1JSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/9L9/KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAPxi8a5/Zn/AG04vEa5ttD1K/XUWPRP7O1bcl1x0xDIZdo/6ZrX7OAhgGU5B5BHQivz2/4KE/DY698PtL+I9hFuuvDNz9nuyo5NjelVyfXy5wgHoJGNe5fskfEsfEz4I6Jd3Uvmanoq/wBj3+TljJaKojc9yZITGxPdifSm9rgfTFFFFIAooooAKKKKACiiigAooooAKKKKACiivJPF3xz+F/gy+/sbU9bju9YJKppOmI+o6iz/AN37PbLJIpP+2FHvQB63RXzv/wAJ/wDHDxjx4D8BxeHbN+U1Lxlc+Q+PbTrQyz59BJJH747H/CmfG/if5/ib8SNav425bTvD6poNjg9UZod91IvbLTAn9KdgPUvFvxJ+H/gOIy+MvEWm6PxuCXd1HHK4/wBiMne59lUmvLv+Gh9O1z5Phr4R8T+Md/Ed1a6e1hp7H3u7826Y91DevSu18JfBP4T+B5RdeGvC+nW92Du+2Sxfabwt6m5nMkxJ93r1KjQD54+3/tOeJubTSvC3gi2fgm+uZ9bvkHqEtxb2+fYyMPr1o/4Up4z1z5vHnxT8Tah6waL5GgWxHdSLZGnI+s2fUmvoeii4Hhmlfs1/BHS7n7dJ4VtdVvOrXOsvLqszH1LXjzc59MY7V7Hpuk6Vo1sLPR7K3sbdekVtEsMY+ioAK0KKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAfib4g/bP/aDtdf1S0tNctIoLe+uYYk/s62bakcrKoyyEnAHU81k/8NrftGf9DBaf+Cy1/wDiK+bvE3/Iz6z/ANhK8/8AR71iV+W5jxFmFLF1acKlkpNLRbJvyPn62NrRqSipaXZ9Wf8ADa37Rn/QwWn/AILLX/4ij/htb9oz/oYLT/wWWv8A8RXynRXH/rPmf/P38I/5Gf1+v/N+R/Qp8AvGGueP/g94Y8YeJZUm1PU7RpLmSNBErOsrpkIuAMhR04zXsFfPH7J3/Ju/gn/rxk/9Hy19D1+vH0gUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFcv4g8EeDPFiGPxToOmawpGMX9nDc8f9tFauoooA+fZv2Y/hRBK914Vt9S8I3bnPn+HdUu9OIP8A1zjk8n846i/4Vn8aPDvPg34oT6hDHyln4p0y31AN7NdW32Wf8fmr6Hop3A+eP+E0/aE8NceJ/AGm+JYF5e78L6qI3C+1pqCxMT7LM349amtv2lvhpbzx2XjMar4JvJDtWHxNp0+nqT7TlXtiPcS4PavoGoLm2tryB7W7iSeGUbXjkUOjA9ipyCPrQBR0bXdD8RWS6j4f1G01O0f7s9nOlxEfo8ZZT+dateG6z+zj8I9TvG1bTNHbw1qp5XUPDlxLpFwrf3v9FaNGP++jVk/8IJ8dvCHzeCfHlv4ltE4TT/GFmGk2/wDYQsRFLn3eJ6APoiivnf8A4Xb4n8K/J8WfAGsaJCv39V0fGu6YFHWR2tgLmJfZ4OPX19V8HfEXwL8QbQ3vgrXbHWI1GXW1mV5I/aSPPmRn2dQaVgOzooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//T/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDnfF3hnTPGnhfVfCWspvstXs5rOcdwkyFdy+jLnKnsQDX5N/sc+KNT+C/x9134LeK38mPV5pNOYNwg1GxZjA65/hmQuq4+8Wj9q/Yavyf/b5+HF/4U8Y6B8dfC+62kuJYLa8niGGh1Cz+e1nz/eeNNueg8pe5prsB+sFFeYfBr4k2Hxa+G2ieOrHar39uBdwqf9ReRfJPH64WQHbnkqQe9en0gCiiigAooooAKKKKACisXxB4j0HwnpM+veJtQttL062XdLc3UixRL6DcxGSegA5J4HNeFD4jfEv4ofufg9oy6LoknH/CV+I4HjSRD/HYad8k0+QcrJMYoz6NQB7n4i8TeHfCOly634o1O10mwh+/cXkywxg9hucgFj2A5PYV4j/wujxT43/c/BTwhdazbvwuv65v0nRgD0eLzFN1dKO/lRAejVteHfgL4SsdUj8UeNZ7rxx4kT5l1PXmW4WBupFragC2tlB5URxhh/eNe4UwPnX/AIUx4u8Z/vvjF42v9Ugk5fQ9A3aLpIU9Y3MTG7uF93mH+7XrnhHwD4K8BWX9neDNDsdGgIAcWcCxtJjvI4G5z7sSfeuuopXAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA/mm8Tf8AIz6z/wBhK8/9HvWJW34m/wCRn1n/ALCV5/6PesSvxPNv9+rf4pfmz5XEfxZerCiiivPMT97f2Tv+Td/BP/XjJ/6Plr6Hr54/ZO/5N38E/wDXjJ/6Plr6Hr98PsAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvKfGPwS+GXjm7Gq6zokUOrKd0eraezWGoxv2YXNuY5SR2DEj2r1aigD51PhD46+Af3ngnxRB430yPppXiseVfBBzti1O3QbnPQefC3u1X9L/aB8N2t/DoPxO06++H2sTN5ccetqosJ37i31GMtayj6ujH+7XvdZ+qaTpeuWE2la1Z29/ZXC7Jra6iWaGRfRkcFWH1FMC7HJHNGs0LK8bgMrKcqynkEEcEEU+vnWX4I6v4JkfUfgX4hl8NfMZG0DUN9/wCH5yeSBCzedaFj1a3cAf3COKt6V8cV0XUoPDPxm0h/BGrXD+VbXksgn0O/ft9nvwFRGbr5U4jcZA5NFgPf6KRWVlDKQQRkEcgg0tIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//U/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigArz/4p/D3S/in4A1rwJq+Fi1S2ZI5SMmC4X54ZQPWOQK2O4GOhr0CigD8hv2K/iLqnwm+Kms/Abxxm0TU7ySCGOQ/LBq9v8hUE4G25RdoP8TLHj71frzX5dft6/Bm7sbyy+PnhBHhnt3gg1loMq8ckZAtbwEcgghYmbPBEeO5r7B/Zp+Nln8b/hta63M6LrunBbPWYFwNtyq8SqvaOdRvXsDuXkqab7gfQlFcTovxE8G+IfFut+BtI1OK41vw6ITqFop+eMTKGUjswGQH2k7GIDYJFdtSAKKKKACvFPGvxeNhrj+Afhzpp8V+MdoMtpFJ5dlpiP8Adm1G5wVhXuIxmV+iryDXO+JPGfij4n+Ib34cfCW7OnWGnSG38R+LEAcWb/xWWn5+WS9x9+TlLcHnMmAPXfA/gPwv8O9DTw/4Usxa24YyzSMTJcXM78vNcStl5ZXPLOxJ7DAAAAPNvDvwVF7q0HjP4v6iPGXiOFvMto5Y/L0fTGP8NlZElAy9POl3ytgHKniveaKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooA/mm8Tf8AIz6z/wBhK8/9HvWJW34m/wCRn1n/ALCV5/6PesSvxPNv9+rf4pfmz5XEfxZerCiiivPMT97f2Tv+Td/BP/XjJ/6Plr6Hr54/ZO/5N38E/wDXjJ/6Plr6Hr98PsAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKztX0fSdf0240fXLODULG6Qxz21zGssMinsyMCpH1FaNFAHzXJ8P/H3wgY3/AMHJ21zw4nzTeDdTuCTEnf8Asq8kJaEjtBKWiPOChxXqvw/+Jfhf4kWE9zoMssN5Yv5Oo6Zexm31DT5+8VzA3zI3BweVbGVYiu/rx/4ifCiHxRfweM/CV6fDfjbTk22esQLlZoxz9mvYuBc2znqrfMh+ZCDwWB7BRXkXw0+J0viu5vfCHi2xGg+NdEVTqWllt0csTHCXlm5/11pKfut95G+R8HBPrtIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD/9X9/KKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDM1rRtL8RaRe6Drdul3YahBJbXMEgykkUqlWU/UH61+JniGL4jfsPfGW/bw232rStUtZxYPcgm3vrN8+X5gXAM1rIV3YwcjsknP7iO6RI0kjBEQFmZjgADqSewFfgV+1h8cG+NHxLmk0uYv4b0HzLLSVB+WUZHm3OPWdlBHT92qAjINNAeR+Cvin4z8C/EGH4maRfO+tLdPc3MkxLC789i06TjI3rLk7u+TkYYAj+gb4RfFXw38Y/BFl418NPhJx5d1aswMtpdIB5kMmO65yDgblIYcGv5tK+h/2b/j1q/wACfG6an+8ufD+pFINYsVOfMiB+WaMHjzockr/eBKkgNkU0B/QjXgXxX8S6/r2u2PwV8A3b2Ws61AbvWNUh+/o2ihtjzKe1zcNmK3HY7n42g165ZeK/Duo+F08aWF9FcaLLZnUEvEOYzbBDIX9RhQcg8jBBGa8g/Z60y7v/AA1e/FfXYyutfEK5/tmXfy0GnEbdOtgf7kVrtYf7TtUgeveEvCegeB/Dtj4V8MWiWWm6fEIoYk/NmYnlndiWdjksxJPJroqKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFAH803ib/kZ9Z/7CV5/6PesStvxN/wAjPrP/AGErz/0e9Ylfiebf79W/xS/NnyuI/iy9WFFFFeeYn72/snf8m7+Cf+vGT/0fLX0PXzx+yd/ybv4J/wCvGT/0fLXwl+1D+0L8aPBXx08SeFPCfii40zStPWw+z20UFuyp51nBK/zPEzHLux5J646V+5YzGUsLSdas7RR9ZVqxpx5pbH66UV+Bv/DVn7RX/Q8Xn/gPaf8Axmj/AIas/aK/6Hi8/wDAe0/+M143+tmW/wAz+5nL/aNDufvlRX4G/wDDVn7RX/Q8Xn/gPaf/ABmv0j/Yn+Jnjr4m+Bdf1Hx5q0mr3VlrH2eCaWOKNliMET7f3SICNxJ5BPNd2AzrCYybp0JXaV9rG1HFU6rtBn2fRRRXqnQFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAeRfFf4c3Xi+0s/EnhO4TTPGnh1mudD1Ej5d5Hz2txjl7W5HySL2yGHIwdv4YfEC1+JHhSHXUt3sL+CWSx1XTpT+9sNRtjsuLd/dG5U/xIVbvXoVfOd9H/AMK5/aA0+/tv3ejfE21ks7xBwkeuaXGZYJvQG4tQ8Zx95o1J5pgfRlFFFIAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP/W/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAa6LIpRwGVgQQRkEHqCK8g8Tfs+/BPxhvbX/AAXo8skn35obZbWds+stv5ch/wC+q9hooA+GPE3/AAT7+CGsb5NCn1jQJD9xbe5FxCPqtwkkhH/bQfWvnnxN/wAE3vFdvvfwd4w0+/HVY9StpbMj23xG4BPvtFfrbRTuwPyQ0/w1+0Z+z98GfHfgLxjpAvfCWp6XcxWuo2d5DOmnXF1+6f5C4mEM4chh5eFchuMua/WDStOt9H0uz0izULBY28VtEo4ASJQij8AK86+OHhmfxj8H/GHhuzQyXN5o92LdB1aeOMyRL+MiqK6P4eeKLfxt4E8P+LrZw6avptrdkjs8sas6n3VsqR2IoYHY0UUUgCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP5pvE3/ACM+s/8AYSvP/R71iVt+Jv8AkZ9Z/wCwlef+j3rEr8Tzb/fq3+KX5s+VxH8WXqwooorzzE/e39k7/k3fwT/14yf+j5a/K39sn/k5Xxh/u6X/AOm+2r9Uv2Tv+Td/BP8A14yf+j5a/K39sn/k5Xxh/u6X/wCm+2r9b4q/5Fs/l+aPo8w/gP5HzJRRRX5IfOBX61/8E7P+Sc+Kv+w+P/SWGvyUr9a/+Cdn/JOfFX/YfH/pLDX2PBX++T/w/qj08r/iP0P0Jooor9LPdCiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACvnz9o4fZvC/hjXk/12i+M/Dt5F65e8S3cf8CjmdT7GvoOvnr48t/bF/wDDvwHB89xrni6wu5I/Wx0bdf3DfQeUi/8AAhTQH0LRRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD/1/38ooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAK+bfhdL/wAKx8e6z8E9R/da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UaJpbj0a3sdkjj/fmb3r6IoouBxvhD4eeBfANr9j8F6Dp+jRkYc2lukckn/XSQDe592JNdlRRSAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAP//T/fyiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigDn9c8JeFfE8fk+JNG0/Vo8Y231rFcrj0xIrCvJLz9mL4HXFw15Z+GI9IuW5E2j3NzpjKfUC0liXj0wRXvdFAHz1/woXUNN+fwp8S/G2lsv3I7jUY9Ut19vLvYZmwPTfR/whn7RelfPpfxI0nWsdItZ8PJFn/eksriH9EFfQtFO4Hz2NX/AGoNJ+a98OeDfECjtp+qXmnSH/gNzbzKP++zR/wtf4tad83iD4QawsY6vpOq6dqP5KZYHP4qK+hKKAPnv/honSrbnW/BHjvSFH3nufD08sa/V7Uzr+RpV/am+BasI7/xG2mSHgpqWn31iQfQme3jH619B0jKrqVYAg8EHkGgDyXT/j38EdUwLPx54dZm6K+p28Tn/gMjq36V29h4y8IaoAdM1zTbwHp5F3DLn/vljUWoeB/BWrZ/tXw/pd7nr9osoZc/99Ia4m+/Z++Buokm58BeHcnqYtNghJ/GNFNGgHrysrqGUhgehHINLXz637LHwD3FoPCcNoT3s7u7tD+cE6V5F40+Eng7wsXHh59bsdmdvleINWGMfW8NAH2/RX43+M/F/jjwyH/sXxb4lgCZ2htc1CYDHtLcMK+f739pb49addGO18c6wFHQST+b/wChhqfKB/QhRX8+8H7XX7Rtv/q/G12f9+3tZP8A0OE1pRftn/tLRkBfGbH/AHtO09v52po5WB++1FfiJpn7Xv7RFxHum8Wbj/2DdPH8rauntv2rfj5Jjf4oz/3D7D/5Go5WB+ylFfj5J+1L8d1jyPE/P/YPsf8A5HrGuP2sPj+mdvinH/cOsP8A5Go5WB+zlFfhhrH7Y37R1qD5Hi7b/wBw3Tj/ADta5OT9sr9pSX73jOQf7thYL/6DbCjlYH790V/PhP8AtZftE3OfM8b34z/zzjgj/wDQIhWr4f8A2gfjbrUwTUPG+tsGbB8u8kh/9FlaOUD9+qK/KLwZca54nKDXfEniW7D4yreINTVefZboCvqzwr+z/wDC/wARQGXXbPVL9toP7/XtWcE+4N5g/jSA+qp7m2tU8y5lSFP70jBR+Zrk9Q+I/wAPNJz/AGr4o0Wzx1+0ahbxY/76kFebQfsufAC3fzP+EK0+d/71yZbkn6maR811lh8Efg1pZDWHgXw5Cw6ONLti/wD30Yy360gMTUP2j/gNpmftPjzQnx/z73sdz/6JL1jf8NQ/CC5+XRLzVdbfsmmaJqVzu+jLbBD+DV7Xp/h3w/pOP7K0yzs8dPs9vHFj/vlRWzT0A+e/+F+3N38uifDTx7fM33Wk0hbGM+5a7nhIHvg0f8LG+O1//wAgn4SG2Rvuyat4hsoMe5jt1um/Dg19CUUAfPW79qfVOieBdAif+82oapOg/AWsefzFH/Ctvjjqf/Ic+LUltG/34dF0GztcD0WW4a6f8eK+haKLgfPX/DOegX/Hizxd408SI334r7XriGBvX91Z/ZkA+grc0X9nT4G6DL9osfBWkyzZz517B9vlz6+ZdGVs++c17TRRcCnY6dp+l262mmWsNpAv3YoI1jQfRVAFXKKKQBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAf//Z)
<!-- #endregion -->

<!-- #region id="0CvYgCZYSDlc" -->
#Example 1: TdTomato and GFP

<!-- #endregion -->

```python id="2SqAibxQ772v"
def apply_filter(em_df, filter_df):
  filter_df = filter_df.set_index('Wavelength')
  combo_df = em_df.join(filter_df, on='Wavelength', how='inner', rsuffix=' Filter')
  return np.dot(combo_df.loc[:,'Emission'], combo_df.loc[:,'Emission Filter'])



def get_A(Filter_list, FP_list):
  A = np.zeros((len(Filter_list), len(FP_list)))
  for i, filter in enumerate(Filter_list):
    filter_df = filter_dict[filter]
    for j, FP in enumerate(FP_list):
      spectra_df = spectra_dict[FP]
      A[i,j] = apply_filter(spectra_df, filter_df)
      
  row_sums = A.sum(axis=0)
  A = A / row_sums[ np.newaxis, :]
  return A


def main_2_by_2(FP_list, Filter_list, x_known, two_channels=False):
  A = get_A(Filter_list, FP_list)

  b, x_inferred, res = mock_unmixing(A, x_known)

  unmixing_plots(A, b, x_known, x_inferred, two_channels=two_channels, label_list=FP_list)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 315} id="rO9s3EtM8THO" outputId="de4a5c70-a7b5-4726-a9d8-ead5e309864c"
FP_list = ['GFP', 'TdTomato']
Filter_list = ['Green Filter', 'Red Filter']

plot_spectra(FP_list, Filter_list)
```

```python id="nyow8iEXg3of" colab={"base_uri": "https://localhost:8080/", "height": 305} outputId="3cb29c73-36b8-440b-905c-bed55fbfec3e"
#@title { run: "auto" }
GFP_amount = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
tdTomato_amount = 0.7 #@param {type:"slider", min:0, max:1, step:0.1}
x_known = np.array([GFP_amount, tdTomato_amount])

main_2_by_2(FP_list, Filter_list, x_known, two_channels=True)
```

<!-- #region id="RRA2SD_wTKkq" -->
#Example 1 conclusions
For this example the effect is essentially cleaning up the red and green signals. This is because the two flourophores are quite seperable, and the filter set is well aligned to seperate them. However this need not be the case. The only requirement is that the unmixing coefficients (columns of the matrix $\textbf{A}$) be linearly independant. 

Note that even when the two signals are easily seperable, without unmixing the seperation is imperfect. If you are using a both an activity indicator (Geci) and a structural marker in tandem to help account for motion artifacts - you should really unmix the signals to make sure that your real activity is not leaking into your static structural channel. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 305} id="7edbVhW0iZxc" outputId="52f47366-75ff-485f-d90a-34e03b325009"
#@title Example 2: two arbitrary flourophores { run: "auto" }
#@markdown ####Flourophore ratio = (emission in channel 2)/(emission in channel 1)

fp_A_ratio = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
fp_B_ratio = 0.1 #@param {type:"slider", min:0, max:1, step:0.1}

A = np.array([[1-fp_A_ratio, fp_A_ratio],[1-fp_B_ratio, fp_B_ratio]])
A = A.T


fp_A_amount = 0.6 #@param {type:"slider", min:0, max:1, step:0.1}
fp_B_amount = 0.9 #@param {type:"slider", min:0, max:1, step:0.1}
x_known = np.array([fp_A_amount, fp_B_amount])

b, x_inferred, res = mock_unmixing(A, x_known)

unmixing_plots(A, b, x_known, x_inferred, two_channels=True)
```

<!-- #region id="IMJEZD1dUxqr" -->
#Example 2 conclusions:
From this example we can see that it is not necessary that the flourophores be highly seperable, or that the filter sets be well aligned, as long as the combination is sufficient to render the unmixing coefficiencts linearly independant. However, be WARNED! the less seperable the flourophores are, the more susceptible the unmixing will be to errors from noise from various sources. If the unmixing coefficients are very similar, then small fluctuations in the detected photons can lead to large changes in the estimated fluorophore amounts. 

At the end of the notebook we will explore possible sources of noise and try to determine when accurate unmixing is possible. 


<!-- #endregion -->

<!-- #region id="-CQVWQAe3vc1" -->
#Increasing the number of effective channels

In the first two examples we looked at examples with two flourophores and two detection channels. If you want to image more than two colors, you need a way to increase the effective number of channels. Having fewer channels than flourophores will mean that the unmixing coefficients are not linearly independant so the the number of effective channels must always be greater than or equal to the number of distinct fluorophores. Lets look at an example with GFP, mOrange and mCherry to see a few examples of how this might work. 

<!-- #endregion -->

```python id="82a5927b" colab={"base_uri": "https://localhost:8080/", "height": 315} outputId="e0e750a2-a3d6-4ffb-fb25-75bf23f9516d"
FP_list = ['GFP', 'mOrange', 'mCherry']
Filter_list = ['Green Filter', 'Red Filter']
plot_spectra(FP_list, filter_dict)

```

<!-- #region id="4Z-3m_J15TU5" -->
#1. Adding an additional PMT
The simplest way to increase the number of channels is to add an additional PMT and filter set. As long as the proportions of flouresence in the new channel are not identical to those in another channel, this will render the unmixing coefficients linearly independant and allow for unmixing.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 315} id="YM0Q53L75p98" outputId="d40323e1-a306-4855-b9f3-aaa38381f1ba"
filter_dict['Far Red Filter'] = create_filter(700, 100, max_em = 100, steepness=10)
color_dict['Far Red Filter'] = 'maroon'
Filter_list = ['Green Filter', 'Red Filter', 'Far Red Filter']
FP_list = ['GFP', 'mOrange', 'mCherry']

plot_spectra(FP_list, Filter_list)

```

```python id="f700a7f6" colab={"base_uri": "https://localhost:8080/", "height": 375} outputId="3daa6941-62c7-4b46-bd3d-79b31add74a7"
#@title { run: "auto" }
GFP_amount = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
mOrange_amount = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
mCherry_amount = 0.7 #@param {type:"slider", min:0, max:1, step:0.1}
x_known = np.array([GFP_amount, mOrange_amount, mCherry_amount])

main_2_by_2(FP_list, Filter_list, A, x_known)
```

<!-- #region id="H13UGZQv-PfR" -->
#2. Sequential imaging
If an additional channel cannot be added, or would not yield sufficient seperability, then you must rely on sequentially acquired images that are registered to eachother, to capitalize on alternative filter sets or excitation wavelengths. 

###Not good for Activity indicators
* It should be noted that this is not really an option if you need to demix in real time, or at the single pixel resolution. For example if you are trying to demix multiple activity indicators (GECI, GEVI, GluSNFR, etc. and/or a structural marker for normalization) then you really need as many or more channels as flourophores. 

###Not good for subcellular resolution
* Furthermore, demixing subcellular structures (like synapses), has only been done using multiple simultaneously acquired channels (Nedivi lab). Its possible that sequential images could be used, if the resolution was high enough and the registration accurate enough to ensure that the same subcellular structures corresponded to the same pixels. 

###Should be sufficient for sizeable, stable structures
* To my knowledge, this has not been shown using in-vivo 2photon, but I am confident that for stable, well defined structures, we should be able to sucessfully mix entire ROIs. Anything that can be consistently tracked and registered from session to session. Somas, beint the largest will be easiest, but I believe boutons will be feasible as well, especially if we use a structural marker that is visible in all sequential imaging sessions. The dream for me is to use multiple colors to identify the nature of presynaptic cells corresponding to each spine on a single neuron. 

**(One possible alternative to sequential imaging is to interleave different excitation wavelengths so that pixels are acquired at nearly the same time which is possible with the Femtonics Atlas...)
<!-- #endregion -->

<!-- #region id="NlirOx3jYnay" -->
#2a. Sequential imaging with different filter sets
If we are limited to 2 PMTs, we can do multiple rounds of imaging seperating the florescence in different ways. We can imagine using dedicated narrow filter sets for each flourophore, and repeating until we have accounted for each flourophore, or we could use broader filter sets that maximize the photons collected but rely more heavily on unmixing. Additional work will be necessary to assess which method to use, but here is proof of concept:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 312} id="0BXiMtvtXpGH" outputId="fd5f4812-aed6-43ce-8fba-bc59ce08521f"
filter_dict['Orange Filter'] = create_filter(575, 50, max_em = 100, steepness=10)
color_dict['Orange Filter'] = 'orange'
filter_dict['FR Filter'] = create_filter(625, 50, max_em = 100, steepness=10)
color_dict['FR Filter'] = 'r'

Filter_list1 = ['Green Filter']
FP_list = ['GFP', 'mOrange', 'mCherry']

fig, axs = plt.subplots(1,3, figsize=(18, 4))
ax = axs[0]
plot_spectra(FP_list, Filter_list1, ax=ax)
ax.set_title('Imaging Session 1')

Filter_list1 = ['Orange Filter']
ax = axs[1]
plot_spectra(FP_list, Filter_list1, ax=ax)
ax.set_title('Imaging Session 2')

Filter_list1 = ['FR Filter']
ax = axs[2]
plot_spectra(FP_list, Filter_list1, ax=ax)
ax.set_title('Imaging Session 3')

```

<!-- #region id="19OAYgWlYn0h" -->
Its easiest to conceptualize using a seperate imaging session for each flourophor as above. This isn't necessary however, so we will also consider the slighlty less laborious scenario that rewuires two imaging sessions and switching one filter. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 295} id="CQx-EpzgZ6MJ" outputId="692c9896-5eea-49db-f06c-dd711084f87b"


Filter_list1 = ['Green Filter', 'Red Filter']
FP_list = ['GFP', 'mOrange', 'mCherry']

fig, axs = plt.subplots(1,2, figsize=(12, 4))
ax = axs[0]
plot_spectra(FP_list, Filter_list1, ax=ax)
Filter_list2 = ['Green Filter', 'Orange Filter']
ax.set_title('Imaging Session 1')
ax = axs[1]
plot_spectra(FP_list, Filter_list2, ax=ax)
ax.set_title('Imaging Session 2')
Filter_list_of_lists = [Filter_list1, Filter_list2]
```

```python id="8dd23ba7" colab={"base_uri": "https://localhost:8080/", "height": 305} outputId="ff6bb742-1525-41a5-a1e6-c5b5032beb32"
#title {run: auto}

GFP_amount = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
mOrange_amount = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
mCherry_amount = 0.2 #@param {type:"slider", min:0, max:1, step:0.1}
x_known = np.array([GFP_amount, mOrange_amount, mCherry_amount])

main(FP_list, Filter_list_of_lists, x_known)
```

<!-- #region id="eOPBDPmHraW6" -->
#2b. Sequantial imaging with different excitation wavelengths
In some cases it may be inconvenient or impossible to change the filter sets. In these cases you can use multiple excitation wavelengths. In some cases this may be necessary to get sufficient excitation from each of the flourophores (i.e. you will likely need to lower the wavelength to excite blue fluorophores). 

The ideal excitiation wavelengths will create the most seperability between the floresence produced. In most cases this will probably mean targeting 1 or more excitation peaks, and taking advantage of the sharp dropoff in excitation at higher wavelengths. Its important to note that the 2P excitation spectra is not simply a doubled version of the 1P excitation spectra. The two photon excitation spectra are typically broader and blue shifted relative to a doubled version of the 1P. For this proof of concept example we will ignore this phenomenon, but we will address it later when assessing noise and realistic feasibility. 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 467} id="twwqHB5eUsVo" outputId="d2bd4df6-4f4b-4459-aafc-c6a0578b3f10"
def get_FP_colors(FP_list):
    cmap = plt.cm.get_cmap('cool')
    colors = []
    for i in np.arange(0,1,1/len(FP_list)):
      colors.append(cmap(i))
    return colors

def get_filter_colors(filter_list):
    cmap = plt.cm.get_cmap('autumn')
    colors = []
    for i in np.arange(0,1,1/len(FP_list))[::-1]:
      colors.append(cmap(i))
    return colors

def plot_ex_em_spectra(FP_list, Filter_set_list=None, Excitation_list=None):
    if not(type(Filter_set_list[0]) == type([])):
      Filter_set_list = [Filter_set_list]
    if not(type(Excitation_list) == type([])):
      Excitation_list = [Excitation_list]
      #print(Filter_list_of_lists)
    num_sessions = max((len(Filter_set_list), len(Excitation_list)))
    plot_rows = 1
    if not(Excitation_list[0] is None):
      plot_rows = 2
    print(num_sessions)
    fig, axs = plt.subplots(plot_rows,num_sessions, figsize=(num_sessions*6, 4*plot_rows), squeeze=False)

    colors = get_FP_colors(FP_list)

    for i in range(num_sessions):
        if not(Excitation_list[0] is None):
          try:
            Excitation = Excitation_list[i]
          except IndexError as E:
            Excitation = Excitation_list[0]

        
        try:
          filter_set = Filter_set_list[i]
        except IndexError as E:
          filter_set = Filter_set_list[0]
        filter_colors = get_filter_colors(filter_set)
        row_num=0
        if not(Excitation_list[0] is None):
            row_num=1
        for k, filter_name in enumerate(filter_set):
            df = filter_dict[filter_name]
            plot_spectrum(
                axs[row_num,i], 
                df.loc[:,'Wavelength'], 
                df.loc[:,'Emission']*100, 
                #label=f'{filter_name} Emisison', 
                color=filter_colors[k], alpha=.1)
                
      #max_emission=0
        max_emission=100*100
        for j, FP in enumerate(FP_list):
              df = spectra_dict[FP]
              scale_by=100
              row_num=0
              if not(Excitation_list[0] is None):
                plot_spectrum(axs[0,i], df.loc[:,'Wavelength'], df.loc[:,'Excitation'], label=f'{FP} Excitation', color=colors[j])
                scale_by = df.set_index('Wavelength').loc[Excitation, 'Excitation']
                row_num=1
              #max_emission = max(max_emission, scale_by)    
              plot_spectrum(axs[row_num,i], df.loc[:,'Wavelength'], df.loc[:,'Emission']*scale_by, label=f'{FP} Emisison', color=colors[j])



        if not(Excitation_list[0] is None):
          df = create_filter(Excitation, 3, max_em=100, steepness=10)
          plot_spectrum(axs[0,i], df.loc[:,'Wavelength'], df.loc[:,'Emission'], label=f'Excitation')
          axs[0,i].legend()
          axs[0,i].set_title(f'Imaging Session {i}')
          axs[0,i].set_xlabel('Wavelength')
          axs[0,i].set_ylabel('Relative Excitation Percentage')

      #adjust the y axis to account for the scaling 
        axs[row_num,i].set_yticks([0,max_emission/2, max_emission])
        axs[row_num,i].set_yticklabels([0,50, 100])
        axs[row_num,i].legend()
        axs[row_num,i].set_xlabel('Wavelength')
        axs[row_num,i].set_ylabel('Relative Emission Percentage')

filter_dict['Broad Filter'] = create_filter(550, 200, max_em = 100, steepness=10)
color_dict['Broad Filter'] = '0'
Filter_list = ['Broad Filter']
FP_list = ['GFP', 'YFP', 'mOrange', 'mCherry']
plot_ex_em_spectra(FP_list, Filter_set_list=Filter_list, Excitation_list=[450, 500, 550, 600])


```

```python colab={"base_uri": "https://localhost:8080/", "height": 305} id="BZ0v-Y13ak6M" outputId="05ff853f-bd39-4aeb-f868-5d60fda0bdb0"
#title {run: auto}
def recursive_len(item):
  if type(item) == list:
      return sum(recursive_len(subitem) for subitem in item)
  else:
      return 1

def get_A(Filter_set_list, FP_list, Excitation_list=None):

  if not(type(Filter_set_list[0]) == type([])):
    Filter_set_list = [Filter_set_list]
  if not(type(Excitation_list) == type([])):
    Excitation_list = [Excitation_list]
    #print(Filter_list_of_lists)
  num_sessions = max((len(Filter_set_list), len(Excitation_list)))
  channels = 0
  for i in range(num_sessions):
    try:
      channels+= len(Filter_set_list[i])
    except IndexError as E:
      channels+= len(Filter_set_list[0])
  #channels = recursive_len(Filter_list_of_lists)
  A = np.zeros((channels, len(FP_list)))
  
  counter=0
  for i in range(num_sessions):
    if not(Excitation_list[0] is None):
      try:
        Excitation = Excitation_list[i]
      except IndexError as E:
        Excitation = Excitation_list[0]

    try:
      filter_set = Filter_set_list[i]
    except IndexError as E:
      filter_set = Filter_set_list[0]

    for filter in filter_set:
      filter_df = filter_dict[filter]
      for j, FP in enumerate(FP_list):
        spectra_df = spectra_dict[FP]
        scale_by=100
        if not(Excitation_list[0] is None):
          scale_by = spectra_df.set_index('Wavelength').loc[Excitation, 'Excitation']
        A[counter,j] = apply_filter(spectra_df, filter_df)*scale_by
      counter+=1

  row_sums = A.sum(axis=0)
  A = A / row_sums[ np.newaxis, :]
  return A


def main(FP_list, Filter_set_list, x_known, Excitation_list=None, two_channels=False):
  A = get_A(Filter_set_list, FP_list, Excitation_list=Excitation_list)

  b, x_inferred, res = mock_unmixing(A, x_known, verbose=False)

  unmixing_plots(A, b, x_known, x_inferred, two_channels=two_channels, label_list=FP_list)



GFP_amount = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
mOrange_amount = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
mCherry_amount = 0.8 #@param {type:"slider", min:0, max:1, step:0.1}
yfp_amount = .7
x_known = np.array([GFP_amount, mOrange_amount, mCherry_amount, yfp_amount])

main(FP_list, Filter_list, x_known, Excitation_list=[450, 500, 550, 600])

```

<!-- #region id="ZD6haG5DMWJT" -->
#Introducing noise
Lets return to a simple 2 flourophore-2 channel example to look at what happens when we introduce noise to the imaging process. We will look at two highly overlapping flourophores, GFP and CFP with a standard red green filter a a baseline far-from-optimal scenario. 


<!-- #endregion -->

```python
Filter_list = ['Green Filter', 'Red Filter']
FP_list = ['CFP', 'GFP']
excitation_wavelength = 420
plot_ex_em_spectra(FP_list, Filter_set_list=Filter_list, Excitation_list=[excitation_wavelength])
```

First lets add some randomness to the photons being produced by the laser - the actual wavelength of photons produced  and the total number of photons will be sampled form a guassian distribution. We do not expect there to be much variance in the power, since other sources of shot noise likely outweigh this one. Also We are not actually trying to account for the full number of realistic photons here, but just enough to introduce reasonable variability. 

```python id="v92raHMtMsNd"
num_photons = 100 #@param {type:"slider", min:0, max:1000, step:10}

wavelength_sd = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
power_sd =  0.5 #@param {type:"slider", min:0, max:1, step:0.1}

CFP_amount = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
GFP_amount =  0.5 #@param {type:"slider", min:0, max:1, step:0.1}

amount_dict = {'CFP':CFP_amount, 'GFP': GFP_amount }


def plot_photons(realistic_photons,perfect_photons=None, title = 'Excitation photons'):
    fig, ax = plt.subplots()
    wave_min = int(min(realistic_photons))-1
    wave_max = int(max(realistic_photons))+1
    bins = np.arange(wave_min, wave_max, 1)+.5
    
    width = 0.35


    values, edges = np.histogram(realistic_photons, bins=bins, density=True)
    rects1 = ax.bar(edges[:-1] - width/2+.5, values, width, color='0', alpha=.5)

    if not(type(perfect_photons) is None):
        if type(perfect_photons) ==pd.DataFrame:
            plot_spectrum(ax, perfect_photons.loc[:,'Wavelength'], perfect_photons.loc[:,'Emission']/100, color='0', hatch='///')
            legend_elements = [Patch(hatch='///', edgecolor='0',
                     label='Perfect photons', facecolor='0', alpha=.5),
               Line2D([0], [0], color='0', lw=2, label='Realistic photons')]
        else:
            values, edges = np.histogram(perfect_photons, bins=bins, density=True)
            rects2 = ax.bar(edges[:-1] + width/2+.5, values, width, hatch='///', color='0', alpha=.5)
            legend_elements = [Patch(hatch='///', edgecolor='0',
                     label='Perfect phtons', facecolor='0', alpha=.5),
               Patch(edgecolor='0',
                     label='Realistic phtons', facecolor='0', alpha=.5)]
    ax.legend(handles=legend_elements)
    ax.set_xlim((min(realistic_photons), max(realistic_photons)))
    ax.set_xlabel('Wavelength')
    if len(bins)<11:
        ax.set_xticks(bins-.5)
    else:
        interval = (wave_max-wave_min)/10
        if interval < 20:
            interval = 10
        if interval >20:
            interval = 50
        xticks = np.arange(round(wave_min,-1), round(wave_max,-1), interval)
    ax.set_title(title)
    ax.set_ylabel('Fraction of photons')
    
def get_photons(num, num_sd, wavelength, w_sd):
    num_photons = np.random.normal(num, num_sd, 1)
    phton_wavelengths = np.random.normal(wavelength, w_sd, int(num_photons))
    return phton_wavelengths
    
perfect_photons = get_photons(num_photons, 0, excitation_wavelength, 0)
realistic_photons = get_photons(num_photons, power_sd, excitation_wavelength, wavelength_sd)
plot_photons(realistic_photons,perfect_photons)
```

Next we need to account for two things to make flourophore excitation more realistic. First the two photon excitation spectra is not identical to the one photon. It is broader and slightly blue shifted. Second, we want the interaction between the flourophores and the photons to be probabilistic. For each photon we draw from a uniform random distrubtion and threshold based on the value of the 2P excitation spectra at that wavelength so that an appropriate fraction of photons sucessfully induce emission. We then sample this number of photons from the emission distribution to determine their wavelength. 

```python
def convert_spectra_to_2P(oneP_spectra_df):
    filter = np.zeros(200)
    filter[:5]=.4
    
    filter[5:65] = .1
    filter[65:70] = 1
    filter[70:]= 0
    filter = filter/np.sum(filter)
    twoP_spectra_df = oneP_spectra_df.copy()
    twoP_spectra_df.loc[:,'Excitation'] = np.convolve(oneP_spectra_df.loc[:,'Excitation'], filter, mode='same')    
    return twoP_spectra_df

def get_excited_flourophores(photons, twoP_spectra_df, FP_amount=1):
    #FP_amount should bebetween 0 and 1 where 1 indicates that all the photons interact with flourophores (unrealistic?)
    excitation_thresholds = twoP_spectra_df.loc[:,'Excitation'][np.searchsorted(twoP_spectra_df.loc[:,'Wavelength'], photons)]
    random_draw = np.random.uniform(0,100,len(excitation_thresholds))
    random_draw2 = np.random.uniform(0,1,len(excitation_thresholds))
    noninteracting_photons = random_draw2>FP_amount
    random_draw[noninteracting_photons] = 100
    
    excited_FPs = len(np.nonzero(random_draw<np.array(excitation_thresholds))[0])
    return excited_FPs

def get_excited_flourophores_nonrandom(photons, twoP_spectra_df, FP_amount=1):
    excitation_thresholds = twoP_spectra_df.loc[:,'Excitation'][np.searchsorted(twoP_spectra_df.loc[:,'Wavelength'], photons)]
    
    return int(np.sum(excitation_thresholds/100))*FP_amount


def get_emission_wavelengths(num_excited, spectra_df):
    #here we need to sample from the probability distribution defined by the spectra, and I'm not sure how to do that
    #could probably come up with something weighting them all approprriately, stack end to end, then numpy uniform
    wavelength_list = []

    for wavelength in spectra_df.loc[:,'Wavelength']:
        num = int(spectra_df.set_index('Wavelength').loc[wavelength, 'Emission'])
        wavelength_list.extend(list(np.ones(num)*wavelength))
    wavelength_list = np.array(wavelength_list)                                                   
    photons_drawn = np.random.randint(0,len(wavelength_list), num_excited)
    return wavelength_list[photons_drawn]

def get_emission_wavelengths_nonrandom(num_excited, spectra_df):
    total = np.sum(spectra_df.loc[:,'Emission'])
    normalized = np.array(spectra_df.loc[:,'Emission'])/total
    fractional_photons = num_excited*(normalized)
    spectra_df = pd.DataFrame({'Wavelength': spectra_df.loc[:,'Wavelength'] , 'Emission': fractional_photons})
    return spectra_df
    
FP_list_2P = []
excited_FP_dict = {}
emission_photons = {}
for FP in FP_list:
    key = f'{FP}_2P'
    spectra_dict[key] = convert_spectra_to_2P(spectra_dict[FP])
    FP_list_2P.append(key)
    emission_photons[FP] = {}
    excited_FP_dict[FP] = {}
    excited_FP_dict[FP]['realistic'] = get_excited_flourophores(realistic_photons, spectra_dict[key])
    excited_FP_dict[FP]['perfect'] = get_excited_flourophores_nonrandom(perfect_photons, spectra_dict[key])
    emission_photons[FP]['realistic'] = get_emission_wavelengths(excited_FP_dict[FP]['realistic'] , spectra_dict[FP])
    emission_photons[FP]['perfect'] = get_emission_wavelengths_nonrandom(excited_FP_dict[FP]['perfect'] , spectra_dict[FP])
    
plot_ex_em_spectra(FP_list_2P, Filter_set_list=Filter_list, Excitation_list=[excitation_wavelength])


#plot the number of photons from each flouropohore
def plot_emission_count(excited_FP_dict):
    perfect = []
    realistic = []
    for fp, count_dict in excited_FP_dict.items():
        perfect.append(count_dict['perfect'])
        realistic.append(count_dict['realistic'])
    fig, ax = plt.subplots()

    edges = np.arange(0,len(realistic),1)
    width = 0.35
    rects1 = ax.bar(edges - width/2, realistic, width=width, color='0', alpha=.5)

    if not(perfect is None):
        rects2 = ax.bar(edges + width/2, perfect, width=width, hatch='///', color='0', alpha=.5)
        legend_elements = [Patch(hatch='///', edgecolor='0',
                     label='Perfect_phtons', facecolor='0', alpha=.5),
               Patch(edgecolor='0',
                     label='Realistic_phtons', facecolor='0', alpha=.5)]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Flourophore')
    ax.set_xticks(edges)
    print(list(excited_FP_dict.keys()))
    ax.set_xticklabels(list(excited_FP_dict.keys()))
    ax.set_title('Number of Photons Emitted From each Flourophore')
    ax.set_ylabel('Photons Emitted')
          
def convert_to_spectra_df(wavelength_list):
    wave_min = min(wavelength_list)-3
    wave_max = max(wavelength_list)+3
    wavelengths = np.arange(wave_min, wave_max, 1)
    counts, bins =  np.histogram(wavelength_list, bins=wavelengths+.5, density=False)
    spectra_df = pd.DataFrame({'Wavelength':  wavelengths[1:], 'Emission': counts})
    return spectra_df

        
plot_emission_count(excited_FP_dict)
                        
#plot the Emissione wavelengths of each flourophore
                        #or all together
for FP, phton_dict in emission_photons.items():
    #print(phton_dict['realistic'])
    #print(type(phton_dict['perfect']))
    #print(type(phton_dict['perfect'])==pd.DataFrame)

    plot_photons(phton_dict['realistic'],phton_dict['perfect'], title = f'{FP} Emission photons')
    #phton_dict['realistic'] = convert_to_spectra_df(phton_dict['realistic'])
                        
```

```python

```

```python
#probably need to similate more losses - 
#Xphotons that don't ineract 
#photons that emit but are then lost

#XNeed to factor in the amount of the flourophores here too - maybe can do that when we factor in photon loss?
#XAmount acounts for photons that don't interact

#Xthen we will need to handl the fact that "perfect" is producing fracitions of photons at each wavelength. 
#Now they are both dataframes with wavelength and emission

    

#then filters
#and PMTs - how to include PMT sensetivity? check hamamatsu again, can't remember what the efficiency represents. 


def apply_filter_stochastic(photons, filter_df, fraction_photons_reaching_filter=1):
    #fractions of photons lost is based on the fact that flourophores emit photons in 3 dimensions, 
    #independant of those lost through filter imperfections
    
    pass_through_thresholds = filter_df.loc[:,'Emission'][np.searchsorted(twoP_spectra_df.loc[:,'Wavelength'], photons)]
    random_draw = np.random.uniform(0,100,len(excitation_thresholds))
    random_draw2 = np.random.uniform(0,1,len(excitation_thresholds))
    noninteracting_photons = random_draw2>fraction_photons_reaching_filter
    random_draw[noninteracting_photons] = 100
    
    passed_photons = photons[np.nonzero(random_draw<np.array(pass_through_thresholds))[0]]
    return passed_photons

#Have to figure out how to preserve wavelength information for the "perfect version" for the PMT? or maybe just skip this?
#yea, lets just skip this and use the same thing we were doing before
def create_PMT():
    #making these curves manually - imput a few values and then interpolate 
    annotated_wavelengths = [300, 350, 450, 500, 550, 600, 700, 760]
    quantum_efficiency = [9, 25, 41, 42, 43, 42, 30, 0]
    wavelengths = np.arange(min(annotated_wavelengths), max(annotated_wavelengths), 1)
    quantum_efficiency = np.interp(wavelengths,annotated_wavelengths, quantum_efficiency)
    spectra_df =  pd.DataFrame({'Wavelength':  wavelengths, 'Emission': quantum_efficiency})
    return spectra_df


def apply_PMT_stochastic(photons, filter_df, fraction_photons_reaching_filter=1):
    return apply_filter_stochastic(photons, filter_df, fraction_photons_reaching_filter)


#parameters for filters
Filter_max_transmission = 0.5 #@param {type:"slider", min:0, max:1, step:0.1}
filter_steepness =  5 #@param {type:"slider", min:0, max:10, step:1}

filter_wavelenght_list = [500, 600]
bandwidth = 100
perfect_filter_list = []
realistic_filter_list = []
for i, wavelength in enumerate(filter_wavelenght_list):
    key = f'Filter {i}'
    prefect_key = f'{key} perfect'
    perfect_filter_list.append(key)

    realistic_key = f'{key} realistic'
    realistic_filter_list.append(key)
    filter_dict[realistic_key] = create_filter(wavelength, bandwidth, max_em=Filter_max_transmission, steepness=(2**a)+.1)
    filter_dict[prefect_key] = create_filter(wavelength, bandwidth, max_em=100, steepness=10000)

plot_ex_em_spectra(FP_list, Filter_set_list=perfect_filter_list)
plot_ex_em_spectra(FP_list, Filter_set_list=realistic_filter_list)
raise()


#combined_photons = 
#photons_after_filters = 
#plot

#PMT = create_PMT()


#detected_photons = 

#plot
```

```python
a = np.arange(0,20,1)
(2**a)+.1
```

```python

```

```python

Then need to include bleaching for multiple images

produce example unmixing plots - raw scatter plot, unmixing vect and hist around mean at different points. 

then produce a summary from lots of combinations of flourophore possibilitie


```

```python
print(spectra_dict)
```

```python id="AOfcxHhjXdJ5"
 #at some point we should hide all this plotting code in plotting and move it up so that it auto updates

def plot_unmixing_ratios_2vec(A, ax, label_list=None):
  #2 indicates that this is only intended to work when it is passed 2 channels as in the first 2-3 examples. 
  N, C = A.shape
  numbered_channels = np.arange(0,C,1)
  colors = get_FP_colors([1,2])
  if label_list is None:
    label_list = ['FP1', 'FP2']
  ax.quiver([0], [0], A[0,0], A[1,0], color=colors[0], alpha=.5, angles='xy', scale_units='xy', scale=1, label=label_list[0])
  ax.quiver([0], [0], A[0,1], A[1,1], color=colors[1], alpha=.5, angles='xy', scale_units='xy', scale=1, label=label_list[1])

  ax.set_title('Unmixing ratios')
  ax.set_xlabel('Channel 1')
  ax.set_ylabel('channel 2')
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  ax.legend()
  return ax

def plot_unmixing_ratios_bar(A, ax, label_list=None):
  C, N = A.shape
  colors = get_FP_colors(np.zeros(N))
  numbered_channels = np.arange(0,C,1)
  #colors = plt.cm.rainbow(np.linspace(0, 1, N))
  if label_list is None:
    label_list = [f'fp{n}' for n in range(N)]
  for n in range(N):
    ax.plot(numbered_channels, A[:,n], label=label_list[n], alpha=1, color=colors[n])
  ax.set_title('Unmixing ratios')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Percent Flouresence')
  ax.legend()
  ax.set_xticks(numbered_channels)
  return ax


def plot_flourescence_vals(b, ax):
  C = b.shape[0]
  numbered_channels = np.arange(0,C,1)
  ax.bar(numbered_channels, b, color='0')
  ax.set_title('Detected flourescnece values')
  ax.set_xlabel('Channel')
  ax.set_ylabel('Photon count')
  ax.set_xticks(numbered_channels)
  return ax

def plot_flour_proportion(x, A, ax, label_list=None):
  C, N = A.shape
  colors = get_FP_colors(np.zeros(N))
  numbered_channels = np.arange(0,C,1)
  #colors = plt.cm.rainbow(np.linspace(0, 1, N))
  bottom = np.zeros(C)
  if label_list is None:
    label_list = [f'fp{n}' for n in range(N)]
  for n in range(N):
    proportions = x[n]*A[:,n]
    ax.bar(numbered_channels, proportions, label=label_list[n], alpha=.5, bottom=bottom, color=colors[n])
    bottom = bottom+proportions
  ax.set_xlabel('Channel')
  ax.set_xticks(numbered_channels)
  ax.set_title('Proportion of flourescence for FP')
  ax.set_ylabel('Photon count')
  ax.legend()
  return ax

#plot a inferred flourophore amounts 
def plot_flourophore_vals(x_known, x_inferred, ax, label_list=None):
  N = x_known.shape[0]
  x = np.arange(0,N,1)
  colors = get_FP_colors(np.zeros(N))
  width = 0.35
  if label_list is None:
    label_list = [f'fp{n}' for n in range(N)]
  for flour in range(N):
    flour_b = np.zeros(N)
    flour_b[flour] = x_inferred[flour]
    #ax.bar(label_list, flour_b, alpha=.5, label=label_list[flour])
    rects1 = ax.bar(x - width/2, flour_b, width, alpha=.5, color=colors[flour])
    flour_b[flour] = x_known[flour]
    #color = rects1.patches[0].get_facecolor()
    #color_list.append(color)
    rects2 = ax.bar(x + width/2, flour_b, width, hatch='///', color=colors[flour], alpha=.5)
  legend_elements = [Patch(hatch='///', edgecolor='0',
                         label='True Amount', facecolor='1'),
                   Patch(edgecolor='0',
                         label='Inferred Amount', facecolor='1')]

  ax.legend(handles=legend_elements)
  ax.set_xticks(x)
  ax.set_xticklabels(label_list)
  ax.set_title('Inferred flourophore amounts')
  ax.set_xlabel('Flourophore')
  ax.set_ylabel('Amount')
  #ax.set_xticks(label_list)
  return ax


def unmixing_plots(A, b, x_known, x_inferred, label_list=None, two_channels=False):
  #twoD is a bool. If there are more than two channels then we need to avoid the first plot
  fig, axs = plt.subplots(1,4+int(two_channels), figsize=(15, 3))
  plt.tight_layout()
  #plot the unmixing ratios
  if two_channels:
    ax=axs[0]
    ax = plot_unmixing_ratios_2vec(A, ax, label_list)

  ax=axs[0+int(two_channels)]
  ax = plot_unmixing_ratios_bar(A, ax, label_list)

  #plot the computed floresencs values
  ax=axs[1+int(two_channels)]
  plot_flourescence_vals(b, ax)

  #plot a stacked bar of the inferred flourophore amounts accounting for the floresence values
  ax = axs[2+int(two_channels)]
  plot_flour_proportion(x_inferred, A, ax, label_list)

  ax = axs[3+int(two_channels)]
  plot_flourophore_vals(x_known, x_inferred, ax, label_list)


def mock_unmixing(A, x_known, verbose = False):
  C, N = A.shape

  #Compute the detected flourescence amounts
  b = np.dot(A, x_known)

  #Compute the inferred flourophore amounts
  x_inferred, res, rank, s =np.linalg.lstsq(A,b)

  if verbose:
    print(f'A = {A}')
    print(f'N = {N}, C = {C}')
    print(f'Actual flourophore amounts = {x_known}')
    print(f'Detected flourescence values = {b}')
    print(f'Inferred flourophore amounts = {x_inferred}')
  return b, x_inferred, res
```

```python id="wYd4Znr0pZP7"

```
