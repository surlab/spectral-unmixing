# Spectral Unmixing

Unmix the flourophore composition from multi-channel images. 
Also implements a number of useful steps for you imaging pipeline:
1. gaussian spline smoothing for image visualisation and noise reduction


## Installation instructions

1. A recent Python distribution is required. Installing anaconda and checking the box to add it to your path during the installation is one of the easiest ways to satisfy this requirement.
1. you will also need to install git
1. Open a terminal (or anaconda prompt if anaconda was not added to the path) and type:
```bash
cd ..\documents\code #or similar as relevant for your machine
git clone https://github.com/GreggHeller1/spectral-unmixing.git
cd spectral-unmixing
conda env create -f environment.yml
conda activate spectral-unmixing

```

1. Code should now be installed and the spectral-unmixing environment should still be activated. 
1. Currently functionality is best accessed through a jupyter notebook, although it should be fairly easy to implement a command line tool or a gui. For now:

```bash
cd path/to/spectral-unmixing
cd app_scripts
jupyter Notebook
```

OR:
Edit the approprieate "start_unmixing_app_windows" or "start_unmixing_app_mac" file as needed so that paths are correct (add more detail here)

## Usage instructions
### Input Files:
In the notebook, there are a number of parameters. Most are described inline
Most importantly "open_path" should lead to a directory with a series of I16 files suffixed with _Z0.I16, _Z1.I16, _Z2.I16,... _Z###.I16 OR a tiff file.

Also required are the unmixing coefficients for each of the imaged flourophores. The number of flourophores imaged should not exceed the number of channels in the image. Each flourophore should be imaged individually on the exact imaging system (same laser wavelenght, filtersets and PMTs). The unmixing coefficients are the proportion of flourescence that is distributed to each channel for an average pixel. (At some point we will implement code to produce the coefficients from a path to the calibration images). The coefficients for each flourophore should be different (linearly seperable/independant).

If the unmixing coefficients are entered manually into the config.py file or directly in the notebook then the raw images (as I16s or TIFFs) are all the prerequisites files.


### Usage steps

Jupyter Notebook:
```bash
conda activate spectral-unmixing
cd path/to/spectral-unmixing
cd app_scripts
jupyter Notebook
```
OR double click the approproate "start_unmixing_app_windows" or "start_unmixing_app_mac" file or a shortcut to it


command line:
1. Rename "default_config.py" to "config.py" (First time only)
1. Change the paths and parameters in "config.py" as desired
1. In a terminal with the conda environment active (>>>conda activate spectral-unmixing), run >>>python spectral-unmixing.py

### Output Files:
filename_unmixed_smoothed.tiff (or similar)
The code saves one or more TIFF files depending on the parameters. Output tiffs should have the shape N_flourophores x X_pixels x Y_pixels x Z_frames and should be openeable in imageJ. The filename should include tags to describe the processing steps that were completed (linearization, unmixing and/or smoothing). 

filename_unmixed_smoothed.json (or similar)
For each step an additional file is produced documenting the paramters used to produce the corresponding image. This will not be sufficient to perfectly recreate the raw data, so the raw data should always be kept. 

Note: Currently if a file undergoes the same processing steps with different parameters, the resulting filenames will be the same, and the old files will be written. We can check for this case and add attempt # or timestamps to the filename, but this is not currently implemented.

# Quality Control (QC)

It is important that you QC the results of this code and do not trust it blindly. It should be clear that bleed through is being removed where flourophores contribute a substatial proportion of flourescence to two or more channels. You should verify that the smoothing is appropriate for you use case and the size  of the structures typically imaged, the resolution of the scope and the distance between pixels. 

Beware that the PMTs output is not linear after a certain range. This will compromise the unmixing. Make sure the structures/pixels you are interested in are within the linear range, or if you attempt to linearize the PMT signal the range that can be adequately corrected back to linear.

# Credit

This code was created for the surlab at MIT by Gregg Heller. It was created using images provided by Kendyll Burnell and Josiah Boivin and partially based off of code by previous members of the Nedivi lab _____ and ______ (strictly the reading of the I16s and the application of the smoothing). 


