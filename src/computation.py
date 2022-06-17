# +
import matplotlib.pyplot as plt
import os
from skimage import io
import glob
import numpy as np



paths =  [r"/Users/Gregg/Dropbox (MIT)/Files for Gregg/FromJoe/4color_testing/unmixingcoeffs/YFP_915/10us_dwell_915nm"]
#[r"/Users/Gregg/Dropbox (MIT)/Files for Gregg/FromJoe/4color_testing/unmixingcoeffs",
#    r"/Users/Gregg/Dropbox (MIT)/Files for Gregg/Gregg_2p_Imaging"]

subdir_name = 'unmixing_plots'
number_of_hists = 10


def compute_PMT_nonlinearity(chanX, chanY, current_data_dir, label):
    min_lin_val = 4
    max_lin_val = 14
    xs = []
    ys = []
    # loop through until you hit nonlinearity. Compute what linear should be
    #lets choose valid range is 4:14
    for x in range(min_lin_val,max_lin_val):
        y = np.mean(chanY[chanX==x])
        if (y < max_lin_val) and (y > min_lin_val):
            #we want to normalize the vector to put them all on the same playing field
            length = np.linalg.norm(np.array([x,y]))
            xs.append(x/length)
            ys.append(y/length)
    #now average all the vectors in the linear range.
    print('XXXXXXXLin')
    print(chanX)
    print(chanY)
    print(xs)
    print(ys)
    xs_per_y = np.mean(xs)/np.mean(ys)
    print(xs_per_y)
    #Maybe we should try to visualize this somehow?
    fig, ax = plt.subplots()
    ax.scatter(xs, ys)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel('Channel I')
    ax.set_ylabel('Channel J')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Valid unmixing vectors for '+str(label))
    filename = 'Valid unmixing for'+str(label)
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig.savefig(pathname)

    #now loop through and count the number of photons
    detected_photons = []
    true_photons = []
    max_x = np.max(chanX)
    max_y = np.max(chanY)
    if max_y>max_x:
        print("switching axis")
        Chan_temp = chanX
        chanX = chanY
        chanY = Chan_temp
        y = np.mean(chanY[chanX==x])
        xs_per_y = 1/xs_per_y
    for x in range(min_lin_val, np.max((max_x, max_y))):
        if x< max_lin_val:
            detected_photons.append(x)
            true_photons.append(x)
        else:
            y = np.mean(chanY[chanX==x])
            if y>x:
                print("switching axis")
                Chan_temp = chanX
                chanX = chanY
                chanY = Chan_temp
                y = np.mean(chanY[chanX==x])
                assert(x>y)
            if y>max_lin_val:
                #if the y value is out of the linear range then we need to correct that first
                #find the
                y = correct_PMT_nonlinearity(y, detected_photons, true_photons)
            print('VVVVVVVV')
            print(x)
            print(y)
            expected_x = xs_per_y*y
            detected_photons.append(x)
            true_photons.append(expected_x)

    #plot measured photons vs actual photons
    fig, ax = plt.subplots()
    ax.plot(true_photons, detected_photons)
    ax.set_xlabel('true_photons')
    ax.set_ylabel('detected_photons')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('PMT nonlinearity')
    filename = 'PMT nonlinearity from '+str(label)
    subdir = os.path.join(current_data_dir, subdir_name)
    if not(os.path.isdir(subdir)):
        os.makedirs(subdir)
    pathname = os.path.join(subdir, filename)
    fig.savefig(pathname)


def correct_PMT_nonlinearity(photons_measured_to_correct, detected_photons_list, true_photons_list):
    print('############')
    print(photons_measured_to_correct)
    print(detected_photons_list)
    print(true_photons_list)
    try:
        assert(photons_measured_to_correct<np.max(detected_photons_list) and photons_measured_to_correct>np.min(detected_photons_list))
    except AssertionError as E:
        raise(ValueError("the measured photons are not in the correctable range"))
    #Interpolate to account for non-integer "measured photons" from taking hte mean
    dense_ts, dense_trues, dense_detected = dense_interpolate(true_photons_list, detected_photons_list)
    #Find the index of the closest match between the number of detected photons we want to correct and the list of detected photons for which we have true photon values
    num_elements = max(np.shape(dense_detected))
    points = np.tile(photons_measured_to_correct, (num_elements,1))
    diff = dense_detected - points#
    closest_match_idx = np.argmin(diff)
    #take that index from the true photons
    return dense_trues[closest_match_idx]


def distance_along_curve(t_1, t_2, xs, ys):
    #should this just take an array is input instead of the roi style dict?
    dist = 0
    start_t = min(t_1,t_2)
    end_t = max(t_1,t_2)
    assert(end_t<=len(xs))

    #this is where we could introduce direction
    for i in range(start_t,end_t-1):
        x_1 = xs[i]
        x_2 = xs[i+1]
        y_1 = ys[i]
        y_2 = ys[i+1]
        dist += np.sqrt((x_1 - x_2)**2+(y_1-y_2)**2)
    return dist

def dense_interpolate(xs, ys):
    #assuming the coords are in the right order - could be necessary if dendrite loops back near itself
    dense_xs = []
    dense_ys = []
    total_points = 1000 #could parameterize this - maybe 2-3x the pixel dimensions of the image?
    #doing linear interpolation since other methods can be problematic in certain cases like sharp angles. if we are manually annotating this should be fine
    #want to make sure that the density is even along the whole dendrite, not dependant on the density of the original manually annotated points
    #approx_length = int(np.max(xs) - np.min(xs)) + 1
    points_per_distance = 100#int(total_points/approx_length)
    print(xs)
    print(ys)
    for i in range(len(xs)-1):
        seg_xs = xs[i:i+2]
        seg_ys = ys[i:i+2]
        segment_length = np.abs(seg_xs[0]- seg_xs[-1])
        num_points_to_add = 100#round(points_per_distance*segment_length)
        dense_segment_xs = list(np.linspace(seg_xs[0], seg_xs[-1], num_points_to_add))
        dense_xs.extend(dense_segment_xs)
        dense_ys.extend(list(np.linspace(seg_ys[0], seg_ys[-1], num_points_to_add)))
    dense_ts = np.arange(0, len(dense_xs), 1)
    return dense_ts, dense_xs, dense_ys



def main():
    for path in paths:
        for current_data_dir, dirs, files in os.walk(path, topdown=False):
            fig, ax = plt.subplots()
            try:
                for file in files:
                    if 'tif' in file.lower():
                        print(current_data_dir)
                        fullpath = os.path.join(current_data_dir, file)
                        im = io.imread(fullpath)
                        for i in range(im.shape[3]):
                          for j in range(im.shape[3]):
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

            except Exception as E:
                raise(E)
