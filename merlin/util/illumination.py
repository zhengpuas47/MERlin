# Core functions for imaging correction
# by Pu Zheng
# 2024.08

import numpy as np
#import pandas
#import cv2
import time
#from abc import ABC
#from abc import abstractmethod
from scipy.stats import scoreatpercentile
from scipy.ndimage import gaussian_filter


def illumination_correction(
    ims, 
    channels,
    correction_pfs,
    rescale=True,
    verbose=True,
    ):
    """Apply illumination correction for 2d or 3d image with 2d profile (x-y)"""
    _total_illumination_start = time.time()
    if verbose:
        print(f"- Start illumination correction for channels:{channels}.")
    if len(ims) != len(channels):
        raise IndexError(f"length of illumination images and channels doesn't match, exit.")

    # apply correction
    _corrected_ims = []
    for _im, _ch in zip(ims, channels):
        _illumination_time = time.time()
        _channel_profile = correction_pfs[_ch]
        if len(np.shape(_channel_profile)) != 2:
            raise IndexError(f"_channel_profile for illumination should be 2d")
        # apply illumination correction
        if len(np.shape(_im)) == 3:
            _cim = _im.astype(np.float32) / _channel_profile[np.newaxis,:]
        elif len(np.shape(_im)) == 2:
            _cim = _im.astype(np.float32) / _channel_profile
        else:
            raise IndexError(f"input image should be 2d or 3d.")
        # rescale
        if rescale: # (np.max(_im) > _max or np.min(_im) < _min)
            _min,_max = 0, np.iinfo(_im.dtype).max
            _cim = (_cim - np.min(_cim)) / (np.max(_cim) - np.min(_cim)) * _max + _min
            _cim = _cim.astype(_im.dtype)
        else:
            _cim = np.clip(_cim,
                           a_min=np.iinfo(_im.dtype).min, 
                           a_max=np.iinfo(_im.dtype).max).astype(_im.dtype)
        # append
        _corrected_ims.append(_cim.copy())
        del(_cim)
        if verbose:
            print(f"-- corrected illumination for channel {_ch} in {time.time()-_illumination_time:.3f}s.")
    if verbose:
        print(f"- Finished illumination correction in {time.time()-_total_illumination_start:.3f}s.")

    # return 
    return _corrected_ims


def image_2_illumination_profile(
    im, 
    channel,
    remove_cap=True,
    cap_th_per=[5,90],
    gaussian_filter_size=40,
    verbose=True,
):
    """Function to process image into mean-illumination-profile"""
    
    if verbose:
        print(f"-- load image for channel:{channel} ")
        _total_start = time.time()
    # copy image
    _nim = np.array(im, dtype=np.float32)
    # remove extreme values if specified
    if remove_cap:
        _limits = [scoreatpercentile(im, min(cap_th_per)), 
                    scoreatpercentile(im, max(cap_th_per))]
        _nim = np.clip(im, min(_limits), max(_limits))
    _pf = gaussian_filter(np.sum(_nim, axis=0), gaussian_filter_size)
    if verbose:
        print(f"into profile in {time.time()-_total_start:.2f}s.")
    
    return _pf 
