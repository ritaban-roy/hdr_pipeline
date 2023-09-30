# hdr_pipeline
HDR processing pipeline build as part of Computational Photography course @ CMU

## Instructions

There are 4 python files : <br>
1. `hdr_pipeline.py` - Contains the main code for everything except Noise calibration
2. `opts.py` - Contains options for `hdr_pipeline.py`
3. `noise_calibration.py` - Contains code for dark frame calculation, gain and additive noise estimation and finally, optimal weight based merging
4. `capture.py` - gphoto2 capture scripts run by subprocess calls

### hdr_pipeline.py

The help attributes in `opts.py` explain all the options. By default only `--img_dir` needs to be given which is the path to your image stack, and the extension of the images via `--ext`.<br>

```
$python hdr_pipeline.py --img_dir [path_to_img_stack] --ext [tiff or jpg]
```
### noise_calibration.py

Here, the functions `calculate_dark_frame()` and `check_ramp()` need to be run first in that order. At the end of `check_ramp` execution, the gain and additive noise values will be printed respectively. Those values need to be given to `merging()` function through the `gain` and `var_add` argument.<br>
Additionally the **g curve** needs to be passed as usual if the merging is being done for JPEG images<br>

So to summarise :
1. Call `calculate_dark_frame()`
2. Call `check_ramp()` - This will also show all the histogram and plots one by one
3. Call `merging` - One example of a merging call is as follows : `merging(image_dir, ext='tiff', merge_algo='linear', gain=24.98, var_add=9852.45, g=None)`<br> 
`g` should be calculated as done in `hdr_pipeline.py` in case of JPG images

