# +
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

from src import config as cfg


def fp_from_tiffname(filename):
    idx = filename.lower().find("fp")
    return filename[idx - 1 : idx + 2]


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
                            true_photons_list):

    try:
        valid_photons = np.logical_and(
                                       photons_measured_to_correct <= np.max(detected_photons_list),
                                       photons_measured_to_correct >= 0)
        assert valid_photons.all() #np.min(detected_photons_list)).all()

    except AssertionError as E:
        print(valid_photons)
        raise (ValueError("some measured photons are not in the correctable range"))

    # We want to make sure it starts at 0,0, otherwise this will cause strange behavior.
    # Should be linear through the first few points anyway
    detected_photons_list[0] = 0
    true_photons_list[0] = 0


    # linearly interpolate to between the two adjacent points and return the y value
    # use searchsorted to find where this measured photons would be placed (its a mean so won't be an int)
    idxs = np.searchsorted(detected_photons_list, photons_measured_to_correct) - 1

    # compute the fraction of the distance travelled to the next x point
    x_diff = detected_photons_list[idxs + 1] - detected_photons_list[idxs]

    frac = (photons_measured_to_correct - detected_photons_list[idxs]) / (x_diff)

    # and go the same distance to the corresponding next y point
    y_diff = true_photons_list[idxs + 1] - true_photons_list[idxs]
    y = true_photons_list[idxs] + frac * y_diff

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


def linearize_image(image_path):
    pass

