from skimage import io
import numpy as np
import os
from functools import partial
from matplotlib import pyplot as plt
from cp_hw2 import writeHDR, readHDR, read_colorchecker_gm, lRGB2XYZ, xyY_to_XYZ, XYZ2lRGB
import pickle
import math
import gc
import glob
from scipy.stats import linregress

def weighting_scheme_numpy(x, t_k, gain, var_add, zmin=0.05, zmax=0.95):
    #x = X.copy()
    def uniform_func(z):
        if zmin <= z <= zmax:
            return (t_k*t_k)/(gain*z + var_add)
        else: 
            return 0.0
    apply_func = np.vectorize(uniform_func)
    #print(np.max(x))
    return apply_func(x)
    #print(ret1.dtype)
    #print(np.max(ret1))
    #exit()
    

def merging(image_dir, ext='tiff', merge_algo='linear', g=None):
    with open('dark_frame.pkl', 'rb') as pkl:
        dark_frame = pickle.load(pkl)*6
    
    print(np.max(dark_frame))
    
    for k in range(1, 17):
        t_k = (2**(k-1))/2048
        #print(np.max(image_k))
        image_k = io.imread(os.path.join(image_dir, f'exposure{k}.{ext}')) - (t_k * dark_frame)
        #print(np.max(image_k))
        #image_k = image_k 
        if(k == 1):
            I_HDR_num = np.zeros_like(image_k)
            I_HDR_den = np.zeros_like(image_k)
            #underexposed = np.zeros_like(image_k).astype(np.bool_)
            overexposed = np.zeros_like(image_k).astype(np.bool_)
        if(ext != 'jpg'):
            I_k = image_k/(2**16 - 1)
            #underexposed = np.logical_or(underexposed, (I_k > 0.05))
            overexposed = np.logical_or(overexposed, (I_k < 0.95))
        else:
            #print(np.max(image_k))
            I_k = image_k/255
            #underexposed = np.logical_or(underexposed, (I_k > 0.05))
            overexposed = np.logical_or(overexposed, (I_k < 0.95))
            #print("Applying!")
            image_k = np.exp(g[image_k])
        
        
        #print("ik before", np.max(I_k))
        print("Weight apply")
        w = np.zeros_like(I_k)
        #print(np.max(weighting_scheme_numpy(I_k[:, :, 0], t_k=t_k, gain=26.25, var_add=0)))
        #exit()
        w[:, :, 0] = weighting_scheme_numpy(I_k[:, :, 0], t_k=t_k, gain=24.98, var_add=9852.45)
        w[:, :, 1] = weighting_scheme_numpy(I_k[:, :, 1], t_k=t_k, gain=24.98, var_add=9852.45)
        w[:, :, 2] = weighting_scheme_numpy(I_k[:, :, 2], t_k=t_k, gain=24.98, var_add=9852.45)
        print("Done Weights")
        #print(np.max(w))
        if(merge_algo == 'log'):
            try:
                #print(t_k)
                I_HDR_num = I_HDR_num + w * (np.log(image_k + 1e-8) - np.log(t_k))
                print(k, np.max(I_HDR_num))
            except:
                print("Error in log", np.log(t_k))
        else:
            I_HDR_num = I_HDR_num + (w * image_k) / t_k
            print(k, np.max(I_HDR_num))
        
        I_HDR_den = I_HDR_den + w
        
        del image_k
        del I_k
        del w
        gc.collect()
    del dark_frame
    gc.collect()
    
    if (ext == 'tiff' or ext == 'jpg'):
        valid_pixels = (I_HDR_den != 0)
        #invalid_pixels = (I_HDR_den == 0)
        I_HDR = I_HDR_num
        I_HDR[valid_pixels] /= I_HDR_den[valid_pixels]
        #I_HDR[invalid_pixels] = np.min(I_HDR[valid_pixels])
        I_HDR[:, :, 0][I_HDR_den[:, :, 0] == 0] = np.min(I_HDR[:, :, 0][I_HDR_den[:, :, 0] != 0])
        I_HDR[:, :, 1][I_HDR_den[:, :, 1] == 0] = np.min(I_HDR[:, :, 1][I_HDR_den[:, :, 1] != 0])
        I_HDR[:, :, 2][I_HDR_den[:, :, 2] == 0] = np.min(I_HDR[:, :, 2][I_HDR_den[:, :, 2] != 0])
        I_HDR[overexposed == 0] = np.max(I_HDR[valid_pixels])
    else:
        I_HDR_den[I_HDR_den == 0] = 1
        #I_HDR_num[underexposed == 0] = 0
        I_HDR_num[overexposed == 0] = 0
        I_HDR_num[overexposed == 0] = np.max(I_HDR_num)
        I_HDR = I_HDR_num / I_HDR_den
    
    
    if(merge_algo == 'log'):
        I_HDR = np.exp(I_HDR)
    #exit()
    return I_HDR

def calculate_dark_frame():
    dark_files = glob.glob('../data/dark_frames/*.tiff')
    for i, dark_file in enumerate(dark_files):
        curr_image = io.imread(dark_file)
        if(i == 0):
            tot_image = curr_image
        else:
            tot_image += curr_image
    
    tot_image = tot_image/50
    with open('dark_frame.pkl', 'wb') as pkl:
        pickle.dump(tot_image, pkl)

def check_ramp():
    with open('dark_frame.pkl', 'rb') as pkl:
        dark_frame = pickle.load(pkl)
    hist_1 = []
    hist_2 = []
    hist_3 = []
    print(dark_frame.shape)
    for i in range(1, 51):
        curr_image = io.imread(f'../data/ramp_frames/exposure{i}.tiff')
        curr_image = curr_image - dark_frame
        #center = curr_image.shape[0]//2, curr_image.shape[1]//2
        #curr_image = curr_image[center[0]-200:center[0]+200, center[1]-200:center[1]+200, :]
        if(i == 1):
            tot_image = curr_image
        else:
            tot_image += curr_image
        hist_1.append(curr_image[1000][1000][0])
        hist_2.append(curr_image[2000][2000][0])
        hist_3.append(curr_image[3000][3000][0])
    
    
    x = range(1, 51)
    # plt.bar(x, hist_1, width=0.8)
    # plt.show()
    # plt.bar(x, hist_2, width=0.8)
    # plt.show()
    # plt.bar(x, hist_3, width=0.8)
    # plt.show()
    mean_image = tot_image/50
    
    for i in range(1, 51):
        curr_image = io.imread(f'../data/ramp_frames/exposure{i}.tiff')
        curr_image = curr_image - dark_frame
        if(i == 1):
            var = np.square(curr_image - mean_image)/49
        else:
            var += (np.square(curr_image - mean_image)/49)
    
    del dark_frame
    gc.collect()
    mean_image = mean_image.flatten()
    mean_image_rounded = np.round(mean_image)
    sorted_mean_idx = np.argsort(mean_image_rounded)[:65000000]
    
    sorted_means = mean_image_rounded[sorted_mean_idx]
    sorted_vars = var.flatten()[sorted_mean_idx]
    
    print('sorted')
    
    unique_x, inverse_indices = np.unique(sorted_means, return_inverse=True)

    # Use bincount to calculate the sum of corresponding elements in y for each unique element in x
    sum_y = np.bincount(inverse_indices, weights=sorted_vars)
    
    del sorted_means, sorted_vars, mean_image, mean_image_rounded
    gc.collect()
    count_x = np.bincount(inverse_indices)

    # Calculate the average by dividing the sum by the count (avoiding division by zero)
    result_y = sum_y / count_x
    print(unique_x.shape, result_y.shape)
    
    #m, b = np.polyfit(result_y[150:], unique_x[150:], 1)
    #unique_x = unique_x[150:]
    ##result_y = result_y[150:]
    m, b, r_value, p_value, std_err = linregress(unique_x, result_y)
    print(m, b)
    #0.059136328531279966 172.81243958348
    plt.plot(unique_x, result_y, color='blue')
    plt.plot(unique_x, m*unique_x + b, color='red')
    plt.show()
    #print(m, b)
    
    # plt.scatter(range(mean_image.shape[0]), mean_image)
    # plt.show()

if __name__ == '__main__':
    #calculate_dark_frame()
    #check_ramp()
    #6.2e7 26.264379531886387 5245.0600968491635
    #6e7 25.978903316704127 9376.157294784702
    #6.4e7 26.2560571415451 4972.189538328763
    #6.5e7 24.987838304557737 9852.451198679279
    I_HDR = merging('../data/door_stack/', 'tiff', 'linear')
    writeHDR(f'door_noise_linear.hdr', I_HDR)