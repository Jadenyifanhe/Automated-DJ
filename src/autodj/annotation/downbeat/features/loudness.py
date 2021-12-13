import numpy as np
from essentia import *
from essentia.standard import Windowing, Loudness
from sklearn import preprocessing


def feature_allframes(input_features, frame_indexer=None):
    audio = input_features['audio']
    beats = input_features['beats']

    w = Windowing(type='hann')
    loudness = Loudness()

    if frame_indexer is None:
        frame_indexer = range(1, len(beats) - 1)

    loudness_values = np.zeros((len(beats), 1))
    loudness_differences = np.zeros((len(beats), 9))

    # Step 1: Calculate framewise for all output frames
    output_frames = [i for i in range(len(beats)) if (i in frame_indexer) or (i + 1 in frame_indexer)
                     or (i - 1 in frame_indexer) or (i - 2 in frame_indexer) or (i - 3 in frame_indexer)
                     or (i - 4 in frame_indexer) or (i - 5 in frame_indexer) or (i - 6 in frame_indexer)
                     or (i - 7 in frame_indexer) or (i - 8 in frame_indexer)]
    for i in output_frames:
        SAMPLE_RATE = 44100
        start_sample = int(beats[i] * SAMPLE_RATE)
        end_sample = int(beats[i + 1] * SAMPLE_RATE)
        frame = audio[start_sample: end_sample if (start_sample - end_sample) % 2 == 0 else end_sample - 1]
        loudness_values[i] = loudness(w(frame))

    # Step 2: Calculate the cosine distance between the MFCC values
    for i in frame_indexer:
        loudness_differences[i][0] = loudness_values[i] - loudness_values[i - 1]
        loudness_differences[i][1] = loudness_values[i + 1] - loudness_values[i]
        loudness_differences[i][2] = loudness_values[i + 2] - loudness_values[i]
        loudness_differences[i][3] = loudness_values[i + 3] - loudness_values[i]
        loudness_differences[i][4] = loudness_values[i + 4] - loudness_values[i]
        loudness_differences[i][5] = loudness_values[i + 5] - loudness_values[i]
        loudness_differences[i][6] = loudness_values[i + 6] - loudness_values[i]
        loudness_differences[i][7] = loudness_values[i + 7] - loudness_values[i]
        loudness_differences[i][8] = loudness_values[i - 1] - loudness_values[i + 1]

    # Include the raw values as absolute features
    result = loudness_differences[frame_indexer]
    return preprocessing.scale(result)
