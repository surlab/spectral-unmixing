The goal of this project is to develop code tools for a spectral unmixing methods paper, and to generate code to produce reproducible figures for said methods paper. 


demo_figure
Add a feature to the channel definiteion. Currently each channel is defined as an excitation wavelength + a filter. I want to expand this to include any dichroics in the path. This means that on the emission side we need to combine a list of optical elements
for example
Yellow = [535/30, 514-Transmitted]
NarrowGreen = [510/20, 514-Reflected]

As you can see, with the dichroics there is a transmitted and a reflected. If its in the transmitted orientation you can take the transmittance value and multiply by the filter transmission point by point to tget the final trnasmission. If its reflected you need to inver (1-T) for T = transmittance as a decimal value. 

Please update the code to include these channels in the config, to compute the predictions on them appropriately. 

To test, I would like to create a copy of figure5 with a new row dict - call this demo_figure. 
For the specification dictionary
FPS = gcamp ca-, gcamp ca+, yfp, tdtomato, mscarlet, mcherry, mneptune
excitation wavelengths - [800, 920, 1040, 1080, 1180]
filters - narrow green, yellow, orange, red, far red

while you are updating the config, can you also add the additional filters specified for the regular figure 5? 

Then update the test_current bat to produce demo_figure


==========================================

Right now we are working on making figure1 from the images in 
"data/1color_3mice_singleplane_june20250619" I will describe step by step the figure panels that I would like to generate. 

Please make a python file for figure 1
Make a python dictionary to configure the following

Row1_dict = {
    name: excitation based
Fluorophores: [mCherry, mNeptune]
Channel 1: {Excitation wavelength: 1080
    emission filter: [560, 700]}
Channel 2: {Excitation wavelength: 1240
    emission filter: [560, 700]}
}

Row2_dict = {
    name: emission based
Fluorophores: [mCherry, mNeptune]
Channel 1: {Excitation wavelength: 1080
    emission filter: [590, 620]}
Channel 2: {Excitation wavelength: 1080
    emission filter: [645, 695]}
}


Row3_dict = {
    name: dual domain
Fluorophores: [mCherry, mNeptune]
Channel 1: {Excitation wavelength: 1080
    emission filter: [590, 620]}
Channel 2: {Excitation wavelength: 1240
    emission filter: [645, 695]}
}


row_list = [Row1, Row2, Row3]


Please make a seperate function to generate each subpanel, 
a function to generate all subpanels for the configuration of a row
and the main function should generate all subpanels for all rows. 

Subpanel 1:
2P excitation spectra for the 2 fluorophores overlaid with the excitation wavelengths as vertical laser lines. Make sure you are setting the flourophores as function inputs so the function can be reused. Lets set the range from 950-1250 for this (default inputs that can be overridden) plot.ex_em_spectra has some demonstration of how I have made similar plots  in the past. In order to avoid breaking old code please make a new copy of this function to modify. You will need to add functionality for downloading updated spectra, esp the 2P spectra from FPbase

Subpanel 2:
1p emission emission spectra overlaid with the emission filters

Subpanel 3:
table-style visualization of excitation vs emission unmixing. X axis is excitation wavelengths (always shows both 1080 and 1240, even if row only uses one), Y axis is filter sets (labeled broad, red, and far red). Table cells corresponding to configured channels are filled with the fluorophore colors.

Subpanel 4:
visualization of predicted unmixing ratios for each channel. I would like to visualize this as a 3 bar graphs (vertically stacked subplots) - x axis of the bar graphs are all channel 1, channel 2. Top bar graph is overlaid both fluorophores (alpha transparency shows where one surpasses the other). Bars should be normalized so the brightest channel for an FP is 1. middle subplot is just the first FP, bottom subplot is just the second FP. Please make this extensible to many FPs and channels. At some point we will also want to be able to sort the channels be advantage to FP1 over the most dissimilarFP. You can sketch this in but no need to finalize right now.

Subpanel 5:
scatterplot of the two fluorophores and appropriate excitation and emission channels taken from data/1color_3mice_singleplane_june20250619. Dots should be colored according to the FP that generated them
we can subsample if 512x512 will be too dense visually

overlay unit vectors estimated in subpanel 4

overlay unit vectors computed from the data each FP (seperate function to produce these - ignore the lowest 20% and highest 5% of Pixels (this should be configurable so we can easily change the upper and lower bounds across all code, not just plotting code, start a global config file), get the mean angle from the middle chunk)

overlay the angle between the two vectors in degrees (as a little labeled  arc between the two vectors close to the origin)

We will probably want to scale each of these up so they look appropriate compared to the data - maybe they should reach to the 70th percentile of both flourophores 

also overlay classification cones - for the "middle chunk" that we defined, angles should be chosen so that 95% (also globally configurable) of pixels for that FP falls inside the cone. Make a separate function (not in figure code, somewhere in src that computes the cone. This will have to be generalized to higher dimensions). Cones in this plot are just light shading in the appropriate colors (see below, magenta for mCherry and purple for mNeptune). Shade the area in between gray and label this dual expressing. 

Would like to add a line for predicted variance perpindicular to the actual vector. dashed line with ends (like a capital I) that spans to the predicted 95 percentile base on the independant poisson noise in each channel (will be larger when vector is near 45 and small when near 0 or 90) described between triple quotes (please check my logic):
"""95% noise interval perpendicular to an unmixing vector (single fluorophore)

Definitions:

i indexes detection channels (i = 1 to D)

mu_i = expected detected photons in channel i for this fluorophore

u_i = component of the unit unmixing direction (perpendicular to the decision boundary), normalized so that sum over i of (u_i)^2 = 1

Poisson variance per channel: Var_i = mu_i

Projected variance perpendicular to the unmixing direction:

sigma_perp_sq = sum over i of (u_i)^2 * mu_i

95% perpendicular noise interval:

delta_95_perp = 1.645 * sqrt( sigma_perp_sq )
"""

Lets also place the seperability score on the plot as text SS = value

Cell means should be overlaid as larger dots, potentially with a different marker - see sams code, we will have to run the ROI extraction algorithm. We want to use the same ROIs across all rows - this should come from the 1080 channel with the broad red filter and the strongest excitation (highest pockels.)

At some point We will likely want to use the XY alignment code that sam uses in his notebooks, but lets wait on this for now. 


Subpanel 6 and 7
colored images of that generated the scatters in subpanel 5. I want to use magenta for mCherry (red/1080 channel) and purple for mNeptune (farred/1240 channel) - will be two images from each of the two FP sources


Subpanel 8
for each row - 
overlapping (transperant) histogram of angles between all the pixels and each of the vectors (n subplots vertically I think. ) if labels are known color the portion of the bar by the appropriate flourophore 

subpanel 9
9.0
This one combines data across all rows - 
x axis is pixel intensity
y axis is percent correct (incorrect pixels are pixels assigned to the wrong fluorophore, dual expressing or no label)
Lines for Row1, row 2, Row 3. 

9.1 percent correct vs angle seperation
9.2 percent correct vs seperability score
SEPERABIITY SCORE IS:
ok, now lets extend the 95% confidence interval to seperability. I think what we want is - assume N photons produced. Compute the expected N for each channel and each FP. For the FP with fewer photons collected, take that point and find the nearest point on the other FPs vector. SS = 2x distance between these two points/(sum 95% perpindicular confidence interval at these two points). 

9.3 scatterplot of actual angle vs predicted angle
9.4 scatterplot of actual variance vs predicted variance

Answers about 9.2 and 9.4
1. we should start with something large like 10,000 I think. It should be parameterized not hard coded. Although I think this should actually be, assume N flourophores in excitation volume since we need to be able to scale the fractional excitation as well. 2. yes, that function may be useful as long as as this prediction factors in the fractional loss from excitation brightness and emission filters. 3. this should be dimmest in terms of distance from the origin (all dimesnions) (even if this required more photons). Proximity to the origin is what matters for what will be lease decodeable. 9.4 1. yes exactly. 2. Yes, that looks correct. 3. This is a good question. we want the measured from 1 to be comparable to the predicted for 2. If convenient we should work from the same N fps in volume as in 9.2 - it will be good to work on the supporting code for thee together


subpanel 10 (wait on this, requires cellpose)
also combined across all rows - 
same as subpanel 9 but cell based - run the ROI finder from SAMs code (pull this into src), generate means as for subpanel 5. Produce the same curves as subpanel 9. 

======================================================

Figure 1.5
Main figure 
for this figure I want to overlay labels on the scatterplot from dual_domain_subpanel5. No shading for classifiers, and no predicted dashed line. We want labels and arrows demonstrating the following:
Y axis can be stretched or shrunk (thick double ended arrow near the axis maybe from 1500 to 3000) labeled with scaling channel 2 (laser power, PMT amplification, filter collection efficiency)"
The mNEptune vector (another thick double ended arrow near the end of the vector centered somewhere near the middle (1500, 1500)) label this one with scaling pixel brightness (FP concentration, FP brightness, objective collection efficiency, net dwell, ROI size)
Another double headed arrow, perpindicular to the mNeptune vector further out from the previos, maybe (2500, 2000) variance around mean angle (total collection fraction, angle of vector)
Finally for the angle label instead of the actual meausrement label with angle of seperation (emission filter and excitation wavelengths)
somwehre near the origin and detector noise, background and dark noise (a few arrows of differnet lengths and directions away from the origin (can even be negative))

Supplement panels
Then we want to demonstrate examples of each phenomenon. 
1.5a example of same  ch2 filter set with different power compared to the same Ch1. two scatter plots, one for each Ch2. plot all valid paired scatterplots (place in subdir, 2 subplots per figure) and I will select. I think we will have to use the fig2 data, this is where we have same setup, only different pockels
1.5b We want to plot scatterplot of Varance around mean vs angle (FP angle, not seperation angle) for many pairs, colored by the FP, variance taken at a consistent distance from the origin
1.5c Also line plots of variance around mean vs distance from origin along vector
1.5d 1.5 with each line normalied to its own mean variance
1.5e plots of percent correct vs ROI size (subselected within cells)
1.5f plot of dwell time vs percent correct (data in fig3/change_Averaging)
1.5g two scatterplots, for example 800 and 1080 red vs 800 and 1080 broad red - neptune and cherry, ideally this will have the same power in mW...not sure if we have that data. Might have it in Fig 2 data



======================================================

Figure 5 description
fig_5_row_dict = {
    name: fig_5
Fluorophores: [EBFP, tagBFP, mTFP1, gfp, LSSmOrange, TdTomato, mCherry , LSSmKAte, mNetpune]
Excitation wavelengths: 750, 800, 870, 1040, 1180, 1240
emission filters: [400,440][445,475][475,495][500,550][550,580][590,620][645,695]
*We will probably want to optimize excitation strength to maximize the angles. 

subpanel 1 2p excitation spectra for many flourophores from FPbase (see fig 5 row dict)
sunpanel 2 emission spectra for many FPS, with the filters

Subpanel 3 visualization of predicted unmixing ratios for all channels (like subpanel 4 fig 1 - histogram/bar charts)

Subpanel 4 T.Sne of simulated pixels in high dimensions

Subpanel 5 Angle plots showing for each flourophore (like fig 1 subpanel 8 except we can't visualize it all in ONE plot because there are many dimensions. We need 1 plot for each FP. The thing plotted is the angle relative to the Targe FP mean. We want to do a stacked histogram for all the non-taget FPs, and overlap with the target FP
Same plotting code used fot fig 234 subpanel 7


======================================================



Figure 2 description
fig_234_row_dict = {
    name: fig_5
Fluorophores: [GCamp, TdTomato, mCherry, mNetpune]
Excitation wavelengths: [800, 1040, 1180, 1240]
emission filters: [500,550][550,580][590,620][645,695]

Subpanel 0 - diagram from illustrator 

Subpanel 1 - excitation spectra, same as fig 1 Subpanel 1 but add tdtomato, 800, 1040 and 1180 excitatio n and an orange filter (550-580). Make a new “row dict” for this figure 2. 

Fig 5 best chan row dict
ch1: 1040, orange = [550,580]
ch2: 1180, red = [590,620] poc365
ch3: 1240, far red = [645,695], poc594

Subpanel 2 - emission soectrspectra, same as fig 1 but for the new row dict. Include all combinations of excitation and emission. Will want to sort by preference for lowest emission peak FP vs highest emission peak FP

Subpanel 2.1 - emition ratios like fig 5.3

Subpanel 2.2 - emission ratios but instead of all combos
Just use the row dict with these 3 - orange filter, 1040 nm, red filter, 1180 nm, far red filter, 1240 nm (like fg53a)

Subpanel 3 - actua vs theoretical angle scatterplot (similar to fig 1 but includes Tdtom. We want to color by the flourophore.

3.2
is there a way to do this for higher d combos? I guess we compute the angle between

Subpanel 4 - 3 color image overlay based on row dict 2.2  Orange/1040 is colored yellow, 1180/red is colored red and 1240/far red is blue. You will have to make max projections of the stacks

Subpanel 5 3d scatterplot of image in Subpanel 4. Use same scatterplot binning as from figure 1. And we want the cones at some point

Subpanel 6 - triangle projection of the same points in Subpanel 5. Origin is the center of the triangle. Dot location becomes 3d pixel vector times [(0, 1), (cos(30), -1*sin(30)), (-1*cos(30), -1*sin(30)] = [(0,1)(.866, .5), (-.866, .5)]

Subpanel 7 orerlapping histogram of angles between all the pixels and each of the vectors (n subplots vertically I think. ) if labels are known color the portion of the bar by the appropriate flourophore. see figure 5.5


Subpanel 8 - agreement of pixel classifications
8.0 table highlighting the channel subsets. will be similar to fig 1 subpanel 3 except for all the channels here and the highlight color is based on grouping, not FP. lets use shades of blue to avoid confusion. no need to label the cells with channel numbers
8.1
Do the same plot with percent agreement vs brightness for different subsets. for figure 1 it was percent correct. For this one do percent agreement with Best rowdict.
different subsets are - 
subset 1 - all 20mW broad
subset 2 - best row dict
subset 3 1080 +orange, red, far red (emission based)
subset 4 800nm red, far red, 1040 red, far red

Need cell pose classifications for the rest (wait on these)
Subpanel 9 two pie charts for distribution of cell and pixel classifications

Subpanel 10 mean brightness, variance of brightness and variance of angle of pixels in rois vs random shifts of the rois

Subpanel 11 
Agreement of cell classification for different combinations of flourophores. Not sure if using all channels as ground truth will be good? 
Goal is to have some match 100% and the have it fall off a bit but pixels will never match 100% 
Do the same plot with percent agreement vs brightness but with cells this time? Bigger bins?

Need to make “row dicts” for the subsets we want to use 

Subpanel 12 - wavelength and filter table with colors highlighting the different filter subsets for Subpanel 11


Fig 3 adds gcamp to spectra
most of the same plots as figure 2 with the fig 3 data. use the same gcamp ca+ and gcamp ca- process as Figure 5. 






e




