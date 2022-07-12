# +
from matplotlib import pyplot as plt
import numpy as np

def plot_pmt_nonlinearity(true_photons, detected_photons, plot=True):
    #plot measured photons vs actual photons
    fig, ax = plt.subplots()
    ax.plot(true_photons, detected_photons)
    ax.set_xlabel('true_photons')
    ax.set_ylabel('detected_photons')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('PMT nonlinearity')
    if plot:
        plt.show()
    return fig, ax


def plot_channels_im(im, channel_i, channel_j):    
    plot_channels(im[:,:,:,channel_i], im[:,:,:,channel_j], channel_i, channel_j)
    
def plot_channels(x, y, i, j, plot=True, alpha=.01, label=''):
    fig1, ax1 = plt.subplots()
    ax1.scatter(x, y , alpha=alpha)
    ax1.set_xlabel('Channel '+str(i+1))
    ax1.set_ylabel('Channel '+str(j+1))
    title = f"Ch{i} vs Ch{j} for {label}"
    ax1.set_title(str(title))
    ax1.set_aspect('equal', adjustable='box')
    if plot:
        plt.show()
    return fig1, ax1, title

def plot_unmixing_vectors(xs, ys, channel_i, channel_j, label='', plot=True):
    fig, ax = plt.subplots()
    ax.scatter(xs, ys, color='b')
    ax.scatter(np.mean(xs), np.mean(ys), marker='+', color='r', s=50)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(f'Channel {channel_i+1}')
    ax.set_ylabel(f'Channel {channel_j+1}')
    ax.set_aspect('equal', adjustable='box')
    title = 'Valid unmixing ratio for '+str(label)
    ax.set_title(title)
    if plot:
        plt.show()
    return fig, ax, title


def random_plotting_code():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    ax1.scatter(im[:,:,:,i], im[:,:,:,j], alpha= .01)
    if i<j:
        label = 'Ch '+str(i+1)+' vs Ch'+str(j+1)
        try:
            compute_PMT_nonlinearity(im[:,:,:,i], im[:,:,:,j], current_data_dir, label)
        except Exception as E:
            pass
        ax.scatter(im[:,:,:,i], im[:,:,:,j], alpha= .01, label=label)
    ax1.set_xlabel('Channel '+str(i+1))
    ax1.set_ylabel('Channel '+str(j+1))
    filename = 'Ch '+str(i+1)+' vs Ch'+str(j+1)+'.png'
    ax1.set_title(str(filename))
    ax1.set_aspect('equal', adjustable='box')
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig1.savefig(pathname)


    for k in range(number_of_hists):
        max_x = np.max(im[:,:,:,i])
        at_x = int((max_x/number_of_hists)*k)
        values = im[:,:,:,j][im[:,:,:,i]==at_x]
        label = 'Channel'+str(i)+' = '+str(at_x)
        y, bins, patches = plt.hist(values, bins=100, density=True, label=label)
        x = bins[1:]
        ax2.plot(x,y)
    ax2.set_xlabel('Channel'+str(j)+' value')
    ax2.set_ylabel('Percentage of pixels')
    ax2.set_yscale('log')
    ax2.set_title('Intensity spread in Channel'+str(j)+' for fixed Channel'+str(i))
    ax2.legend()
    filename = 'Spread in'+str(j)+' for fixed '+str(i)
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig2.savefig(pathname)


    for x in range(max_x):
        values = im[:,:,:,j][im[:,:,:,i]==at_x]
        values = values - np.mean(values)
        y, bins, patches = plt.hist(values, bins=100, density=True)
        x = bins[1:]
        ax3.plot(x,y)
    ax3.set_xlabel('Channel'+str(j)+' dist from mean')
    ax3.set_ylabel('Percentage of pixels')
    ax3.set_yscale('log')
    ax3.set_title('Spread around mean in Channel'+str(j)+' for fixed Channel'+str(i))
    filename = 'Spread aroudn mean in'+str(j)+' for fixed '+str(i)
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig3.savefig(pathname)

    ax.set_xlabel('Channel I')
    ax.set_ylabel('Channel J')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('All relationships for '+str(filename))
    ax.legend()
    ax.legend.set_alpha(1)
    filename = 'All relationships'
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig.savefig(pathname)
