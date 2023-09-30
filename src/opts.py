import argparse

def get_opts():
    parser = argparse.ArgumentParser(description='15-663 assign1')
    
    parser.add_argument('--ext', type=str, default='tiff',
                        help='JPG or TIFF stack?')
    parser.add_argument('--img_dir', type=str, default='../data/door_stack/',
                        help='directory of LDR image stack')
    parser.add_argument('--wbal', type=bool, default=False,
                        help='set to True to perform white balancing after color correction')

    parser.add_argument('--weight_algo', type=str, default='photon',
                        help='Pixel weighting scheme : [uniform, linear, gaussian, photon]')
    parser.add_argument('--merge_algo', type=str, default='linear',
                        help='HDR merging : [linear, log]')
    parser.add_argument('--color_correction', type=bool, default=True,
                        help='set to True to perform color correction')
    parser.add_argument('--tonemap', type=str, default='xyy',
                        help='Tonemap algo : xyy or rgb')
    parser.add_argument('--key', type=float, default=0.10,
                        help='key for tonemapping')
    parser.add_argument('--burn', type=float, default=0.90,
                        help='burn for tonemapping')
    opts = parser.parse_args()

    return opts