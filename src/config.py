# +
min_lin_val = 10
max_lin_val = 25
# It seems that if these are set too low you get bad results because of noise in the unmixing goefficients.
# You should have a bunch of points in agreement. If there aren't many or there are outliers you need to adjust these bounds
# even if that means allowing a tiny amount of nonlinearity to creep in. Its better than too much noise, and you really should'nt see it until like 50...
# Its also harder when the ratio is further from 1:1 - fewer points will fall in this gap.

min_points_for_valid_unmixing = 5
# this may also help with the above problem. As long as 1 set is high we can probably believe the other.


# +
resolution = 3
# when correcting a non integer photon count (i.e. from a mean) correct to how many decimals
# Since the curve depends on these values errors will compound for higher photon counts

# for the spread plot how many pixels plus/minus do you want to see?
spread_limit = 50
spread_interval = 15
combine_counts = 3
spread_bin_size = 3

figure_path = ("scripts", "results", "figures")
results_path = ("scripts", "results",)
valid_curve_json_filename = 'valid_curves.json'

#path pieces to use for the correction
master_PMT_curve_corrections_suffix = "_mean_2022_08_08_16_16_.npy"
channel_counts = {
                    'ch0': "_ch1_2022_08_08_16_16_.npy",
                    'ch1': "_ch1_2022_08_08_16_16_.npy",
                    'ch2': "_ch1_2022_08_08_16_16_.npy"}

num_channels = 3

