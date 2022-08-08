import pandas as pd
from matplotlib import pyplot as plt

import os
import numpy as np
from src import demo_plotting as plot


def create_filter(wavelength, bandwidth, max_em=100, steepness=10000):
    # steepness is the input to a sigmoid that is used to symettrically shape either end of the filter
    # steepness must be greater than 1
    assert steepness > 1
    range = max(bandwidth, 50)
    wavelengths = np.arange(wavelength - range, wavelength + range, 1)
    up_sigmoid = 1 / (1 + steepness ** (-(wavelengths - (wavelength - bandwidth / 2))))
    down_sigmoid = 1 - (
        1 / (1 + steepness ** (-(wavelengths - (wavelength + bandwidth / 2))))
    )
    emission = np.minimum(up_sigmoid, down_sigmoid)
    emission = emission * max_em
    df = pd.DataFrame({"Wavelength": wavelengths, "Emission": emission})
    return df


def apply_filter(em_df, filter_df):
    filter_df = filter_df.set_index("Wavelength")
    combo_df = em_df.join(filter_df, on="Wavelength", how="inner", rsuffix=" Filter")
    return np.dot(combo_df.loc[:, "Emission"], combo_df.loc[:, "Emission Filter"])


def mock_unmixing(A, x_known, verbose=False):
    C, N = A.shape

    # Compute the detected flourescence amounts
    b = np.dot(A, x_known)

    # Compute the inferred flourophore amounts
    x_inferred, res, rank, s = np.linalg.lstsq(A, b)

    if verbose:
        print(f"A = {A}")
        print(f"N = {N}, C = {C}")
        print(f"Actual flourophore amounts = {x_known}")
        print(f"Detected flourescence values = {b}")
        print(f"Inferred flourophore amounts = {x_inferred}")
    return b, x_inferred, res


# title {run: auto}
def recursive_len(item):
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def get_A(filter_set_list, fp_dict, excitation_list=None):
    if type(filter_set_list) is not list:
        filter_set_list = [filter_set_list]
    if type(excitation_list) is not list:
        excitation_list = [excitation_list]
        # print(Filter_list_of_lists)
    num_sessions = max((len(filter_set_list), len(excitation_list)))
    channels = 0
    for i in range(num_sessions):
        try:
            channels += len(filter_set_list[i])
        except IndexError as E:
            channels += len(filter_set_list[0])
    # channels = recursive_len(Filter_list_of_lists)
    A = np.zeros((channels, len(fp_dict.keys())))

    counter = 0
    for i in range(num_sessions):
        if excitation_list[0] is not None:
            try:
                Excitation = excitation_list[i]
            except IndexError as E:
                Excitation = excitation_list[0]

        try:
            filter_set = filter_set_list[i]
        except IndexError as E:
            filter_set = filter_set_list[0]

        for filter, filter_df in filter_set.items():
            # filter_df = filter_dict[filter]
            for j, (fp, spectra_df) in enumerate(fp_dict.items()):
                # spectra_df = spectra_dict[fp]
                scale_by = 100
                if not (excitation_list[0] is None):
                    scale_by = spectra_df.set_index("Wavelength").loc[
                        Excitation, "Excitation"
                    ]
                A[counter, j] = apply_filter(spectra_df, filter_df) * scale_by
            counter += 1

    row_sums = A.sum(axis=0)
    A = A / row_sums[np.newaxis, :]
    return A


def main(fp_dict, filter_set_list, x_known, excitation_list=None, two_channels=False):
    A = get_A(filter_set_list, fp_dict, excitation_list=excitation_list)

    b, x_inferred, res = mock_unmixing(A, x_known, verbose=False)

    plot.unmixing_plots(
        A,
        b,
        x_known,
        x_inferred,
        two_channels=two_channels,
        label_list=list(fp_dict.keys()),
    )


def convert_spectra_to_2P(oneP_spectra_df):
    filter = np.zeros(200)
    filter[:5] = 0.4

    filter[5:65] = 0.1
    filter[65:70] = 1
    filter[70:] = 0
    filter = filter / np.sum(filter)
    twoP_spectra_df = oneP_spectra_df.copy()
    twoP_spectra_df.loc[:, "Excitation"] = np.convolve(
        oneP_spectra_df.loc[:, "Excitation"], filter, mode="same"
    )
    return twoP_spectra_df


def get_photons(num, num_sd, wavelength, w_sd):
    num_photons = np.random.normal(num, num_sd, 1)
    phton_wavelengths = np.random.normal(wavelength, w_sd, int(num_photons))
    return phton_wavelengths


def get_excited_flourophores(photons, twoP_spectra_df, fp_amount=1):
    # fp_amount should bebetween 0 and 1 where 1 indicates that all the photons interact with flourophores (unrealistic?)
    excitation_thresholds = twoP_spectra_df.loc[:, "Excitation"][
        np.searchsorted(twoP_spectra_df.loc[:, "Wavelength"], photons)
    ]
    random_draw = np.random.uniform(0, 100, len(excitation_thresholds))
    random_draw2 = np.random.uniform(0, 1, len(excitation_thresholds))
    noninteracting_photons = random_draw2 > fp_amount
    random_draw[noninteracting_photons] = 100

    excited_fps = len(np.nonzero(random_draw < np.array(excitation_thresholds))[0])
    return excited_fps


def get_excited_flourophores_nonrandom(photons, twoP_spectra_df, fp_amount=1):
    excitation_thresholds = twoP_spectra_df.loc[:, "Excitation"][
        np.searchsorted(twoP_spectra_df.loc[:, "Wavelength"], photons)
    ]

    return int(np.sum(excitation_thresholds / 100)) * fp_amount


def get_emission_wavelengths(num_excited, spectra_df):
    # here we need to sample from the probability distribution defined by the spectra, and I'm not sure how to do that
    # could probably come up with something weighting them all approprriately, stack end to end, then numpy uniform
    wavelength_list = []

    for wavelength in spectra_df.loc[:, "Wavelength"]:
        num = int(spectra_df.set_index("Wavelength").loc[wavelength, "Emission"])
        wavelength_list.extend(list(np.ones(num) * wavelength))
    wavelength_list = np.array(wavelength_list)
    photons_drawn = np.random.randint(0, len(wavelength_list), num_excited)
    return wavelength_list[photons_drawn]


def get_emission_wavelengths_nonrandom(num_excited, spectra_df):
    total = np.sum(spectra_df.loc[:, "Emission"])
    normalized = np.array(spectra_df.loc[:, "Emission"]) / total
    fractional_photons = num_excited * (normalized)
    spectra_df = pd.DataFrame(
        {"Wavelength": spectra_df.loc[:, "Wavelength"], "Emission": fractional_photons}
    )
    return spectra_df


def model_fp_excitation(fp_dict, realistic_photons, perfect_photons=None):
    # fp_list_2p = []
    spectra_dict_2p = {}
    excited_fp_dict = {}
    emission_photons = {}
    for fp, spectra_df in fp_dict.items():
        key = f"{fp}_2P"
        spectra_dict_2p[key] = convert_spectra_to_2P(spectra_df)
        # fp_list_2p.append(key)
        emission_photons[fp] = {}
        excited_fp_dict[fp] = {}
        excited_fp_dict[fp]["realistic"] = get_excited_flourophores(
            realistic_photons, spectra_df
        )
        excited_fp_dict[fp]["perfect"] = get_excited_flourophores_nonrandom(
            perfect_photons, spectra_df
        )
        emission_photons[fp]["realistic"] = get_emission_wavelengths(
            excited_fp_dict[fp]["realistic"], spectra_df
        )
        emission_photons[fp]["perfect"] = get_emission_wavelengths_nonrandom(
            excited_fp_dict[fp]["perfect"], spectra_df
        )
    return excited_fp_dict, emission_photons, spectra_dict_2p


def apply_filter_stochastic(photons, filter_df, fraction_photons_reaching_filter=1):
    # fractions of photons lost is based on the fact that flourophores emit photons in 3 dimensions,
    # independant of those lost through filter imperfections
    print(photons)
    print(filter_df.loc[:, "Wavelength"])
    rint("here")
    threshold_indicies = np.searchsorted(filter_df.loc[:, "Wavelength"], photons)
    print(threshold_indicies)
    print(filter_df.loc[:, "Emission"])
    pass_through_thresholds = filter_df.loc[:, "Emission"][threshold_indicies]

    random_draw = np.random.uniform(0, 100, len(pass_through_thresholds))
    random_draw2 = np.random.uniform(0, 1, len(pass_through_thresholds))
    noninteracting_photons = random_draw2 > fraction_photons_reaching_filter
    random_draw[noninteracting_photons] = 100

    passed_photons = photons[
        np.nonzero(random_draw < np.array(pass_through_thresholds))[0]
    ]
    return passed_photons


def create_filters(
    filter_wavelenght_list, bandwidth, filter_max_transmission=100, filter_steepness=10
):
    perfect_filters = {}
    realistic_filters = {}
    for i, wavelength in enumerate(filter_wavelenght_list):
        key = f"Filter {i}"
        prefect_key = f"{key} perfect"
        realistic_key = f"{key} realistic"
        realistic_filters[realistic_key] = create_filter(
            wavelength,
            bandwidth,
            max_em=filter_max_transmission,
            steepness=((2**filter_steepness) + 0.1),
        )
        perfect_filters[prefect_key] = create_filter(
            wavelength, bandwidth, max_em=100, steepness=10000
        )
    return realistic_filters, perfect_filters


def combine_photons(emission_photons):
    combined_photons = []
    for i, (fp, photon_dict) in enumerate(emission_photons.items()):
        combined_photons.extend(list(photon_dict["realistic"]))
        try:
            combined_fractions.join(
                photon_dict["perfect"], on="Wavelength", how="Outer", rsuffix=f"{_i}"
            )
        except Exception as E:
            combined_fractions = photon_dict["perfect"].set_index("Wavelength")
    sum_cols = [col for col in combined_fractions.columns if "Emission" in col]
    combined_spectra_df = pd.DataFrame(
        {
            "Wavelength": photon_dict["perfect"]["Wavelength"],
            "Emission": combined_fractions[sum_cols].sum(axis=1),
        }
    )
    return combined_photons, combined_spectra_df


# Have to figure out how to preserve wavelength information for the "perfect version" for the PMT? or maybe just skip this?
# yea, lets just skip this and use the same thing we were doing before
def create_PMT():
    # making these curves manually - imput a few values and then interpolate
    annotated_wavelengths = [100, 200, 300, 350, 450, 500, 550, 600, 700, 760, 1200]
    quantum_efficiency = [0, 0, 9, 25, 41, 42, 43, 42, 30, 0, 0]
    wavelengths = np.arange(min(annotated_wavelengths), max(annotated_wavelengths), 1)
    quantum_efficiency = np.interp(
        wavelengths, annotated_wavelengths, quantum_efficiency
    )
    spectra_df = pd.DataFrame(
        {"Wavelength": wavelengths, "Emission": quantum_efficiency}
    )
    return spectra_df


def apply_PMT_stochastic(photons, pmt_df, fraction_photons_reaching_filter=1):
    return apply_filter_stochastic(photons, pmt_df, fraction_photons_reaching_filter)


def convert_to_spectra_df(wavelength_list):
    wave_min = min(wavelength_list) - 3
    wave_max = max(wavelength_list) + 3
    wavelengths = np.arange(wave_min, wave_max, 1)
    counts, bins = np.histogram(wavelength_list, bins=wavelengths + 0.5, density=False)
    spectra_df = pd.DataFrame({"Wavelength": wavelengths[1:], "Emission": counts})
    return spectra_df
