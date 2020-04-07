from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf

import params
import yamnet as yamnet_model

import os
from keras.models import load_model

def main(argv):
  print("moi")
  yamnet = yamnet_model.yamnet_frames_model(params)
  yamnet.load_weights('yamnet.h5')
  yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')
  

  # Decode the WAV file.
  #wav_data, sr = sf.read(file_name, dtype=np.int16)
  # assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  waveform = argv / 32768.0  # Convert to [-1.0, +1.0]
  sr = 44100
  
  # Convert to mono and the sample rate expected by YAMNet.
  if len(waveform.shape) > 1:
    waveform = np.mean(waveform, axis=1)
  if sr != params.SAMPLE_RATE:
    waveform = resampy.resample(waveform, sr, params.SAMPLE_RATE)

  # Predict YAMNet classes.
  # Second output is log-mel-spectrogram array (used for visualizations).
  # (steps=1 is a work around for Keras batching limitations.)
  scores, _ = yamnet.predict(np.reshape(waveform, [1, -1]), steps=1)
  # Scores is a matrix of (time_frames, num_classes) classifier scores.
  # Average them along time to get an overall classifier output for the clip.
  prediction = np.mean(scores, axis=0)
  # Report the highest-scoring classes and their scores.
  sound_events = np.argsort(prediction)[::-1]
  
  present = []
  prob = prediction[sound_events[0]]
  for event in sound_events:
    event_prob = prediction[event]
    if event_prob >= 0.1:
      present.append(event)

  print('Stream :\n' + 
        '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                  for i in present[:5]))
  return "moi"

'''
if __name__ == '__main__':
  main(['test/5a8910e1.wav'])
'''