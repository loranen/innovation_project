# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:27:53 2020

"""

import numpy as np
from scipy.io import wavfile
import sys

def sliceing():
    fs, signal = wavfile.read('test/cc6ab45b.wav')
    
    slice_length = 4 # in seconds
    overlap = 3 # in seconds
    strating_point = np.arange(0, len(signal)/fs, 
                               slice_length-overlap,
                               dtype=np.int)
    
    if slice_length < overlap:
        sys.exit('Overlap cannot be longer than length of the slice')
    elif slice_length > len(signal):
        sys.exit('Slice cannot be longer than signal')
    
    
    for start in strating_point:
        start_audio = start * fs # Chunk starting index
        end_audio = (start+slice_length) * fs # Chunk ending index
        audio_slice = signal[start_audio: end_audio] # Chunk
    
    return audio_slice
