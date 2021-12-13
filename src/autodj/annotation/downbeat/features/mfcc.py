import numpy as np
from essentia import *
from essentia.standard import Windowing, MelBands, Spectrum
from sklearn import preprocessing

NUMBER_BANDS = 12
NUMBER_COEFF = 5


def feature_allframes(input_features, frame_indexer=None):
    audio = input_features['audio']
    beats = input_features['beats']

    w = Windowing(type='hann')
    spectrum = Spectrum()
    melbands = MelBands(numberBands=NUMBER_BANDS)

    if frame_indexer is None:
        frame_indexer = range(4, len(beats) - 1)

    mfcc_bands = np.zeros((len(beats), NUMBER_BANDS))
    mfcc_bands_diff = np.zeros((len(beats), NUMBER_BANDS * 4))

    # Step 1: Calculate framewise for all output frames
    output_frames = [i for i in range(len(beats)) if (i in frame_indexer) or (i + 1 in frame_indexer)
                     or (i - 1 in frame_indexer) or (i - 2 in frame_indexer) or (i - 3 in frame_indexer)]
    for i in output_frames:
        SAMPLE_RATE = 44100
        start_sample = int(beats[i] * SAMPLE_RATE)
        end_sample = int(beats[i + 1] * SAMPLE_RATE)
        frame = audio[start_sample: end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1]
        bands = melbands(spectrum(w(frame)))
        mfcc_bands[i] = bands

    # Step 2: Calculate the cosine distance between the MFCC values
    for i in frame_indexer:
        mfcc_bands_diff[i][0 * NUMBER_BANDS: 1 * NUMBER_BANDS] = mfcc_bands[i + 1] - mfcc_bands[i]
        mfcc_bands_diff[i][1 * NUMBER_BANDS: 2 * NUMBER_BANDS] = mfcc_bands[i + 2] - mfcc_bands[i]
        mfcc_bands_diff[i][2 * NUMBER_BANDS: 3 * NUMBER_BANDS] = mfcc_bands[i + 3] - mfcc_bands[i]
        mfcc_bands_diff[i][3 * NUMBER_BANDS: 4 * NUMBER_BANDS] = mfcc_bands[i] - mfcc_bands[i - 1]

    # Include the MFCC coefficients as features
    result = mfcc_bands_diff[frame_indexer]
    return preprocessing.scale(result)
