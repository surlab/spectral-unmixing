from src import config as cfg
from src import data_io as io
from src import plotting as plotting
from src import computation as comp



def main(fp, i_number, j_number, channel_i, channel_j, alpha=.01):
    
    #plot and save a scatter plot of the existing pixels
    fig, ax, title = plotting.plot_channels(channel_i, channel_j, i_number, j_number, alpha=alpha, label=fp)
    io.savefig(fig, title)
    
    #plot and save the unmixing ratio that will be used for this image and pair of channels
    xs, ys, xs_per_y = comp.get_unmixing_ratio(channel_i, channel_j)
    fig, ax, title = plotting.plot_unmixing_vectors(xs, ys, i_number, j_number, label=fp, plot=True)
    io.savefig(fig, title)
    
    #compute the PMT correction curve
    detected_photons, true_photons = comp.compute_PMT_nonlinearity(channel_i, channel_j, xs_per_y)

    #Plot and save the inferred nonlinearity 
    fig, ax = plotting.plot_pmt_nonlinearity(true_photons, detected_photons)
    io.savefig(fig, f'PMT curve from {fp} on {i_number}{j_number}')
    print('############')


    
    corrected_i = []
    corrected_j = []
    for i,j in zip(channel_i, channel_j):
        try:
            corrected_i.append(comp.correct_PMT_nonlinearity(i, detected_photons, true_photons))
            corrected_j.append(comp.correct_PMT_nonlinearity(j, detected_photons, true_photons))
        except Exception as E:
            print(i,j)

    fig, ax, title = plotting.plot_channels(corrected_i, corrected_j, i_number, j_number, alpha=alpha, label=f'{fp}_corrected')
    io.savefig(fig, title)
