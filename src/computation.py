# +
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
import logging
from scipy.optimize import nnls


from src import config as cfg
from src import data_io as io

import logging


def fp_from_tiffname(filename):
    idx = filename.lower().find("fp")
    for i in range(1, idx+1):
        if filename[idx - (i)].islower():
            break
    return filename[idx - (i) : idx + 2]


def fake_pmt(actual_photons):
    if actual_photons < 15:
        return actual_photons
    if actual_photons < 300:
        return actual_photons / (1 + (actual_photons - 15) ** 2 / (300 - 15) ** 2)
    else:
        return 150


def reverse_fake_pmt(detected_photons):
    if detected_photons < 15:
        return detected_photons
    if detected_photons < 150:
        return detected_photons * (1 + (detected_photons - 15) ** 2 / (150 - 15) ** 2)
    else:
        return 300


def fake_pmt_n(photons_array, round=True):
    y = []
    x = []
    for actual_photons in photons_array:
        if round:
            y.append(np.round(fake_pmt(actual_photons)))
        else:
            y.append(fake_pmt(actual_photons))
        x.append(reverse_fake_pmt(y[-1]))
    return np.array(x), np.array(y)


def get_valid_pairs(xs_in, ys_in, override_bounds=False):
    xs = []
    ys = []
    # loop through until you hit nonlinearity. Compute what linear should be
    # lets choose valid range from config
    for x in xs_in[np.logical_and(xs_in > cfg.min_lin_val, xs_in < cfg.max_lin_val)]:
        y = np.mean(ys_in[xs_in == x])
        # is there a possibility that this will be unbalanced depending on which is X and which is Y?
        if (
            ((y <= cfg.max_lin_val) and (y >= cfg.min_lin_val)) or (override_bounds)
        ) and not (np.isnan(y)):
            # we want to normalize the vector to put them all on the same playing field
            length = np.linalg.norm(np.array([x, y]))
            xs.append(x / length)
            ys.append(y / length)
    return xs, ys


def get_unmixing_coefs(im_array):
    #identify the axis for color channel
    channel_axis = im_1.shape.index(cfg.num_channels)

    #whats the numpy way to do this? we don't really care if this is fast, but I want to practice thinking this way...
    #mask out all the invalidvalid values

    #maybe we should flaten it first? to just 2 dimensions? channel and value

    #find the channel with the highes max/most variance

    #what if we actually did the division first, then there are no new masks

    #then create new masks I think you have to loop here?
    #No - create an array of all the desired inputs (make sure they are all present in the looping axis

    #pull out the masked values and then average

    #we should now have an array that is channels by X values
    #we divide it by the x value (including the one that has said x values)

    #then take the mean across the values axis


    #im = np.rollaxis(im, channel_axis)
    #for
    return coefs_arrray



def get_unmixing_ratio(xs_in, ys_in):
    xs, ys = get_valid_pairs(xs_in, ys_in, override_bounds=False)
    if len(xs) < cfg.min_points_for_valid_unmixing:
        xs, ys = get_valid_pairs(xs_in, ys_in, override_bounds=True)
        if len(xs) < cfg.min_points_for_valid_unmixing:
            ys, xs = get_valid_pairs(ys_in, xs_in, override_bounds=True)
            if len(xs) < cfg.min_points_for_valid_unmixing:
                raise (
                    ValueError(
                        f"Unable to find enough valid points for unmixing. Found only xs:{xs}, ys:{ys}"
                    )
                )
    xs_per_y = np.mean(xs) / np.mean(ys)
    return xs, ys, xs_per_y


def switch_channels(chanX, chanY, xs_per_y):
    print("switching axis")
    xs_per_y = 1 / xs_per_y
    return chanY, chanX, xs_per_y


def reduce_to_means(chanX, chanY):
    ordered_xs = []
    mean_ys = []
    for x in np.unique(chanX):
        ordered_xs.append(x)
        y = np.mean(chanY[chanX == x])
        mean_ys.append(y)
    return np.array(ordered_xs), np.array(mean_ys)


def polyfit(x, y):
    XX = np.vstack((x**3, x**2, x, np.ones_like(x))).T
    p_no_offset = list(
        np.linalg.lstsq(XX[:, :], y)[0]
    )  # use [0] to just grab the coefs, use XX[:, :-1] to ignore the offset and force through 0
    # now that I'm thinking about it I could probably just set the desired coefficients to 0 (the ones) and still pass them in?and save the hassel of adding back the 0 later??

    # p_no_offset.append(0) #to preserve compatibility with np_polyfit
    return np.array(p_no_offset)


def smooth(y, box_pts_in):
    # code from https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
    box_pts = box_pts_in * 2 + 1  # to ensure an odd number
    box = np.ones(box_pts) / box_pts
    y_smooth = y.copy()
    y_smooth[box_pts_in:-box_pts_in] = np.convolve(y, box, mode="valid")
    return y_smooth


def compute_PMT_nonlinearity(chanX, chanY, xs_per_y):

    # now loop through and count the number of photons
    detected_photons = [0]
    true_photons = [0]

    # the code is designed so that x stays linear longer, otherwise it won't work. If this is not the case we just switch
    # the mapping output is a property of the PMTs, and will be applied to each channel seperately so this is fine
    if np.median(chanY) > np.median(chanX):
        chanX, chanY, xs_per_y = switch_channels(chanX, chanY, xs_per_y)

    for x in np.unique(chanX):
        if x < cfg.max_lin_val / 2:
            # if x is in the linear range, then no correction needs to be computed.
            detected_photons.append(x)
            true_photons.append(x)
        else:
            y = chanY[chanX == x][
                0
            ]  # was taking the mean here, but now do all means in "reduce to means"
            if y > x:
                print(
                    "Warning, the mean seems to have crossed back over - there will be regions where the curve is invalid"
                )
                # x, y, xs_per_y = switch_channels(x, y, xs_per_y)
                # raise(AssertionEror("Switching axis didn't work. You shouldn't hit this code, perhaps the pixel counts are non-monotonic?"))
                # assert(x>y)
            elif y > cfg.max_lin_val:
                # if the y value is out of the linear range then we need to correct that first
                # find the
                y = correct_PMT_nonlinearity(
                    np.array([y]), np.array(detected_photons), np.array(true_photons)
                )
                assert len(y) == 1
                y = y[0]

            expected_x = xs_per_y * y
            detected_photons.append(x)
            true_photons.append(expected_x)
            # print(f'{x}, {y}: {expected_x}')
    return np.array(detected_photons), np.array(true_photons)


def correct_PMT_nonlinearity(
                            photons_measured_to_correct,
                            detected_photons_list,
                            true_photons_list, override=False):
    invalid_photons = np.logical_or(
                               photons_measured_to_correct >= np.max(detected_photons_list),
                               photons_measured_to_correct < 0)
    try:
        invalid_count = np.sum(invalid_photons)
        assert invalid_count==0 #np.min(detected_photons_list)).all()

    except AssertionError as E:
        #print(np.nonzero(invalid_photons))
        err_str = f"{invalid_count} measured pixels are not in the correctable range of photons"
        if not override:
            raise (ValueError(err_str))
        else:
            logging.warning(err_str)
    # We want to make sure it starts at 0,0, otherwise this will cause strange behavior.
    # Should be linear through the first few points anyway
    detected_photons_list[0] = 0
    true_photons_list[0] = 0

    #Add an entry to the end so that all idicies work
    detected_photons_list = np.hstack((detected_photons_list, np.max(photons_measured_to_correct)+1))
    true_photons_list = np.hstack((true_photons_list, true_photons_list[-1]))

    # linearly interpolate to between the two adjacent points and return the y value
    # use searchsorted to find where this measured photons would be placed (its a mean so won't be an int)
    idxs = np.searchsorted(detected_photons_list, photons_measured_to_correct) - 1
    #print(np.max(idxs))
    #print(np.max(detected_photons_list))
    #print()

    # compute the fraction of the distance travelled to the next x point
    x_diff = detected_photons_list[idxs + 1] - detected_photons_list[idxs]

    frac = (photons_measured_to_correct - detected_photons_list[idxs]) / (x_diff)

    # and go the same distance to the corresponding next y point
    y_diff = true_photons_list[idxs + 1] - true_photons_list[idxs]
    y = true_photons_list[idxs] + frac * y_diff
    if override:
        y[invalid_photons] = np.inf
        #could also do this to retrive the originals

    return y


def get_average_curve(curve_dict):


    #Taking the average of these curves is actually a bit tricky since they all have different X coordinates
    xs = np.arange(0,300,1)
    curve_sum = np.zeros(300)
    count=0
    for key, curve in curve_dict.items():
        detected = curve['counts']
        true_photons= curve['corrections']
        #in order to get the correct curve back, we actually need to invert the conversion function
        #by having it correct true photons to detected_photons
        curve_sum += correct_PMT_nonlinearity(xs, true_photons, detected)
        count += 1
    mean_curve = curve_sum/count
    return {'corrections': xs, 'counts': mean_curve}


def linearize_image(im):
    counts, corrections = io.load_master_PMT_curve()
    lin_im = correct_PMT_nonlinearity(
                                im,
                                counts,
                                corrections,
                                override=True)
    return lin_im


def mock_unmixing(A, x_known, verbose=False):
    # Compute the detected flourescence amounts
    b = np.dot(A, x_known)
    b, x_inferred, res = unmix(A, b, verbose)
    if verbose:
        print(f"Actual flourophore amounts = {x_known}")
    return b, x_inferred, res

def test_unmixing_mat(A, verbose=False):
    C, N = A.shape
    if verbose:
        print(f"Unmixing Matrix = ")
        print(F"{A}")
        print(f"Number of flourophores = {N}, Channels = {C}")
    assert C>=N, f"The number of flourophores ({N})must be less than or equal to the number of channels ({C})"


def unmix(A, b, nonnegative=False, verbose=False):
    test_unmixing_mat(A, verbose=verbose)
    C_mat = A.shape[0]
    C_im = b.shape[-1]
    assert C_mat == C_im, f"The number of channels in the image and the unmixing matrix must be the same. Image channels = {C_im}, Unmixing channels = {C_mat}"

    #have to flatten and then reshape b - np.linalg.lstsq only works for 1 and 2 d arrays
    reorder_axis = np.moveaxis(b, -1, 0)
    total_pixels = np.prod(np.array(reorder_axis.shape[1:]))

    channels_x_pixels = reorder_axis.reshape(C_im, total_pixels)

    # Compute the inferred flourophore amounts

    if nonnegative:
        #only positive values - MUCH slower. Consider using numba and implementing yourself if we do this a lot?
        x_inferred = np.empty(channels_x_pixels.shape)
        res = np.empty(channels_x_pixels.shape[1])
        for i in range(channels_x_pixels.shape[1]):
            x, r = nnls(A, channels_x_pixels[:,i])
            x_inferred[:,i] = x
            res[i] = r
    else:
        x_inferred, res, rank, s = np.linalg.lstsq(A, channels_x_pixels)

    if verbose:
        min_val = np.min(x_inferred)
        if min_val<-10:
            logging.warning(f'Some large negative pixel intensities were computed, as low as {min_val}. This may indicate an error in imaging or the coefficient matrix')

    unmixed_reorder_axis = np.reshape(x_inferred, reorder_axis.shape)
    unmixed = np.moveaxis(unmixed_reorder_axis, 0, -1)

    return unmixed, res



