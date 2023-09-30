from skimage import io
import numpy as np
import os
from functools import partial
from matplotlib import pyplot as plt
from cp_hw2 import writeHDR, readHDR, read_colorchecker_gm, lRGB2XYZ, xyY_to_XYZ, XYZ2lRGB
#from cp_exr import writeEXR
import pickle
import math
import gc

def gamma_function(x):
    if x <= 0.0031308:
        return 12.92*x
    else:
        return (1.055*x**(1/2.4) - 0.055)
    
def correct_brightness_and_gamma(rgb_img, scale=1):
    rgb_img = np.clip(rgb_img*scale, a_min=0, a_max=1)
    apply_func = np.vectorize(gamma_function)
    rgb_img[:, :, 0] = apply_func(rgb_img[:, :, 0])
    rgb_img[:, :, 1] = apply_func(rgb_img[:, :, 1])
    rgb_img[:, :, 2] = apply_func(rgb_img[:, :, 2])
    return rgb_img

    
def tonemap_xyz(hdr_image_path, K=0.15, B=0.95, gamma=False):
    I_hdr = readHDR(hdr_image_path)*0.3
    I_xyz = lRGB2XYZ(I_hdr) + 1e-8
    #print(I_xyz.shape)
    x = I_xyz[:, :, 0]/(I_xyz[:, :, 0]+I_xyz[:, :, 1]+I_xyz[:, :, 2])
    y = I_xyz[:, :, 1]/(I_xyz[:, :, 0]+I_xyz[:,:,1]+I_xyz[:, :, 2])
    Y = I_xyz[:, :, 1]
    I_m_hdr = np.exp(np.mean(np.log(Y+1e-8)))
    I_hat_hdr =  K * Y/I_m_hdr
    I_hat_white = B * np.max(I_hat_hdr)
    Y_tm = I_hat_hdr * (1 + (I_hat_hdr/np.square(I_hat_white)))/(1+ I_hat_hdr)
    #print(x.shape, y.shape, Y_tm.shape)
    XYZ_tm  = xyY_to_XYZ(x, y, Y_tm)
    #print(np.dstack(XYZ_tm).shape)
    I_hdr_tm = XYZ2lRGB(np.dstack(XYZ_tm))
    print("Correcting gamma")
    I_hdr_tm = np.clip(I_hdr_tm*0.75, a_min=0, a_max=1)
    gamma_text = ''
    if gamma:
        gamma_text = '_gamma'
        I_hdr_tm = correct_brightness_and_gamma(I_hdr_tm)
    #print(I_hdr_tm.shape)
    #return I_hdr_tm
    hdr_filename = hdr_image_path.split('.')[0]
    io.imsave(f'{hdr_filename}_tmxyz{gamma_text}_brt_075.png', (I_hdr_tm*255).astype(np.uint8))
    
def tonemap_vanilla(hdr_image_path, K=0.15, B=0.95, gamma=True):
    I_hdr = readHDR(hdr_image_path)
    I_m_hdr = np.exp(np.mean(np.log(I_hdr+1e-8)))
    I_hat_hdr =  K * I_hdr/I_m_hdr
    I_hat_white = B * np.max(I_hat_hdr)
    I_tm = I_hat_hdr * (1 + (I_hat_hdr/np.square(I_hat_white)))/(1+ I_hat_hdr)
    gamma_text = ''
    if gamma:
        gamma_text = '_gamma'
        I_tm = correct_brightness_and_gamma(I_tm*0.75)
    print(I_tm.shape)
    hdr_filename = hdr_image_path.split('.')[0]
    io.imsave(f'{hdr_filename}_tm{gamma_text}_0_75.png', (I_tm*255).astype(np.uint8))

def correct_color_correction(hdr_image_fname, ext='jpg', white_balance=False):
    
    with open('color_points.pkl', 'rb') as pkl:
        points = pickle.load(pkl)
    points = np.array(points).astype(np.uint16)
    r, g, b = read_colorchecker_gm()
    color_checker = np.stack([r, g, b], axis=-1).swapaxes(0,1)
    color_checker[[0,1,2,3,4,5]] = color_checker[[5,4,3,2,1,0]]
    color_checker = np.reshape(color_checker, (24, 3))
    
    image_k = readHDR(hdr_image_fname)
    
    patches = []
    for point in points:
        x, y = point
        patch = image_k[y-15:y+15, x-15 : x+15, :]
        patch = np.mean(patch, (0, 1))
        patches.append(patch)
    #patches_stack.append(patches)
    patches = np.array(patches)
    A = []
    B = []
    for i in range(24):
        B.append(color_checker[i][0])
        B.append(color_checker[i][1])
        B.append(color_checker[i][2])
        r = patches[i][0]
        g = patches[i][1]
        b = patches[i][2]
        A.append([r,g,b,1,0,0,0,0,0,0,0,0])
        A.append([0,0,0,0,r,g,b,1,0,0,0,0])
        A.append([0,0,0,0,0,0,0,0,r,g,b,1])
    
    A = np.array(A)
    B = np.array(B).reshape(-1, 1)
    #print(A.shape, B.shape)
    x, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
    affine_matrix = x.reshape(3, 4)
    
    
    hdr_image = image_k
    
    hdr_homo = np.dstack([hdr_image, np.ones((hdr_image.shape[0], hdr_image.shape[1], 1))])
    
    hdr_color = np.zeros((hdr_homo.shape[0], hdr_homo.shape[1], 3))
    
    hdr_color = np.clip(np.matmul(affine_matrix, hdr_homo.reshape(hdr_homo.shape[0]*hdr_homo.shape[1], 4).T), a_min=0, a_max=None)
    hdr_color = hdr_color.reshape((3, hdr_image.shape[0], hdr_image.shape[1]))
    hdr_color = np.moveaxis(hdr_color, 0, -1)
    if white_balance:
        RW, GW, BW = color_checker[-1][0], color_checker[-1][1], color_checker[-1][2]
        hdr_color[:, :, 0] = hdr_color[:, :, 0] * (GW / RW)
        hdr_color[:, :, 2] = hdr_color[:, :, 2] * (GW / BW)
    print("Writing HDR")
    hdr_filename = hdr_image_fname.split('.')[0]
    writeHDR(f'{hdr_filename}_color.hdr', hdr_color)
    return hdr_color, f'{hdr_filename}_color.hdr'

       
def choose_color_points():
    '''
    Call this function to choose the 24 points of color in the image
    '''
    k = 10
    ext = 'tiff'
    test_image = (io.imread(os.path.join('../data/door_stack/', f'exposure{k}.{ext}')))
    #We will take 30x30 patches later, just choose center here
    plt.imshow(test_image)
    points = plt.ginput(n=24, timeout=-1)
    with open('color_points.pkl', 'wb') as pkl:
        pickle.dump(points, pkl)
    

def weighting_scheme_numpy(x, t_k=0, algo='uniform', zmin=0.05, zmax=0.95):
    #x = X.copy()
    if algo == 'uniform':
        def uniform_func(z):
            if zmin <= z <= zmax:
                return 1
            else: return 0
        apply_func = np.vectorize(uniform_func)
        return apply_func(x)
    if algo == 'tent':
        def tent_func(z):
            if(z >= zmin and z <= zmax):
                return min(z, 1-z)
            else:
                return 0
        apply_func = np.vectorize(tent_func)
        return apply_func(x)
    if algo == 'gaussian':
        def gaussian_func(z):
            return (math.e**((z-0.5)**2/0.25)) if (z >= zmin and z <= zmax) else 0
        apply_func = np.vectorize(gaussian_func)
        return apply_func(x)
    if algo == 'photon':
        def photon_func(z):
            return t_k if (z >= zmin and z <= zmax) else 0
        apply_func = np.vectorize(photon_func)
        return apply_func(x) 
    

def merging(image_dir, ext='tiff', weight_algo='uniform', merge_algo='linear', g=None):
    
    
    for k in range(1, 17):
        image_k = io.imread(os.path.join(image_dir, f'exposure{k}.{ext}'))
        if(k == 1):
            I_HDR_num = np.zeros_like(image_k)
            I_HDR_den = np.zeros_like(image_k)
            overexposed = np.zeros_like(image_k).astype(np.bool_)
        if(ext != 'jpg'):
            I_k = image_k/(2**16 - 1)
            overexposed = np.logical_or(overexposed, (I_k < 0.95))
        else:
            I_k = image_k/255
            overexposed = np.logical_or(overexposed, (I_k < 0.95))
            #print("Applying!")
            image_k = np.exp(g[image_k])
        
        t_k = (2**(k-1))/2048
        #print("ik before", np.max(I_k))
        if(weight_algo != 'photon'):
            w = weighting_scheme_numpy(I_k, zmin=0.05, zmax=0.95)
        else:
            w = weighting_scheme_numpy(I_k, t_k=t_k)
        
        if(merge_algo == 'log'):
            try:
                I_HDR_num = I_HDR_num + w * (np.log(image_k + 1e-8) - np.log(t_k))
            except:
                print("Error in log", np.log(t_k))
        else:
            I_HDR_num = I_HDR_num + (w * image_k) / t_k
        
        I_HDR_den = I_HDR_den + w
        
        del image_k
        del I_k
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

def weighting_scheme(x, t_k = 0, algo='uniform', zmin=0.05, zmax=0.95):
    if algo == 'uniform':
        ratio = x/255
        return 1 if (ratio >= zmin and ratio <= zmax) else 0
    
    if algo == 'debevic':
        ratio = x/255
        cutoff = 0.5 * (zmin + zmax)
        if ratio <= cutoff:
            return (ratio-zmin)
        else:
            return (zmax - ratio)
    
    if algo == 'tent':
        ratio = x/255
        return min(ratio, 1-ratio) if (ratio >= zmin and ratio <= zmax) else 0.0
    
    if algo == 'gaussian':
        ratio = x/255
        return (math.e**((ratio-0.5)**2/0.25)) if (ratio >= zmin and ratio <= zmax) else 0.0
    
    if algo == 'photon':
        ratio = x/255
        #print(t_k)
        return t_k if (ratio >= zmin and ratio <= zmax) else 0.0    

def gsolve(Z, B, l, w, t_k=None):
    n= 255
    A = np.zeros((Z.shape[0]*Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))
    
    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if t_k is None:
                wij = w(Z[i][j]+1)
            else:
                wij = w(Z[i][j]+1, t_k[j])
                #print(wij)
            #if wij > 0 : print(wij)
            A[k][Z[i][j]+1] = wij
            A[k][n + i] = -wij
            b[k][0] = wij * B[j] 
            k+=1
    
    A[k][128] = 1
    k+=1
    
    for i in range(n-2):
        if t_k is None:
            A[k][i] = l * w(i+1)
            A[k][i+1] = -2*l*w(i+1)
            A[k][i+2] = l*w(i+1)
        else:
            A[k][i] = l
            A[k][i+1] = -2*l
            A[k][i+2] = l
        
        k+=1
    
    #print(np.sum(A))
    x,resid,rank,s = np.linalg.lstsq(A,b, rcond=None)
    g = x[:n+1]
    lE = x[n+1:]
    #print(np.sum(x), rank)
    return g, lE


def linearize_jpeg(img_dir, algo='uniform', lamda=10):
    img_stack = create_stack(img_dir, 'jpg')
    #print(img_stack.shape)
    num_images = img_stack.shape[0]
    
    ##### Solving for g ##########
    Z = np.reshape(img_stack, (num_images, -1))
    Z = Z.swapaxes(0, 1)
    print(Z.shape)
    w = partial(weighting_scheme, algo=algo, zmin=0.05)
    B = np.log(np.array([(2**(k-1)/2048) for k in range(1, num_images+1)]))
    t_k = None
    if(algo == 'photon'):
       t_k = np.array([(2**(k-1)/2048) for k in range(1, num_images+1)])
    l = lamda
    g, lE = gsolve(Z, B, l, w, t_k)
    g = g.squeeze()
    return g

def read_img_small(image_path):
    jpg_data = io.imread(image_path)
    #print(jpg_data.shape) #4000 x 6000
    #print(np.max(jpg_data), np.min(jpg_data))
    small_data = jpg_data[::200, ::200, :]
    #plt.imshow(small_data)
    #plt.show()
    return small_data



def create_stack(image_dir, ext='jpg'):
    images = []
    for i in range(1, 17):
        images.append(read_img_small(os.path.join(image_dir, f'exposure{i}.{ext}')))
    img_stack = np.array(images)
    return img_stack

def main(opts):
    if(opts.ext == 'jpg'):
        g_func = linearize_jpeg(img_dir='../data/my_stack/', algo='photon', lamda=100)
        plt.plot(g_func)

        plt.xlabel("pixel value")
        plt.ylabel("g")   
        plt.show()
    else:
        g_func = None
    
    I_HDR = merging(opts.img_dir, opts.ext, opts.weight_algo, opts.merge_algo, g_func)
    writeHDR(f'{opts.exp_name}_{opts.weight_algo}_{opts.merge_algo}.hdr', I_HDR)
    curr_file = f'{opts.ext}_{opts.weight_algo}_{opts.merge_algo}.hdr'
    if opts.color_correction:
        I_HDR, curr_file = correct_color_correction(opts.img_dir, f'{opts.exp_name}_{opts.weight_algo}_{opts.merge_algo}.hdr', opts.ext, opts.wbal) 
    
    if(opts.tonemap == 'xyy'):
        tonemap_xyz(curr_file, K=opts.key, B=opts.burn, gamma=True)
    else:
        tonemap_vanilla(curr_file, K=opts.key, B=opts.burn, gamma=True)
    

if __name__ == '__main__':
    from opts import get_opts
    opts = get_opts()
    main(opts)