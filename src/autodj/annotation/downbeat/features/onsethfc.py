import numpy as np
from essentia import *
from sklearn import preprocessing


def feature_allframes(input_features, frame_indexer=None):
    beats = input_features['beats']
    novelty_hwr = input_features['onset_curve']
    HOP_SIZE = 512

    if frame_indexer is None:
        frame_indexer = range(4, len(beats) - 1)

    onset_integrals = np.zeros((2 * len(beats), 1))
    frame_i = (np.array(beats) * 44100.0 / HOP_SIZE).astype('int')
    onset_correlations = np.zeros((len(beats), 21))

    # Step 1: Calculate framewise for all output frames
    output_frames = [i for i in range(len(beats)) if (i in frame_indexer) or (i + 1 in frame_indexer)
                     or (i - 1 in frame_indexer) or (i - 2 in frame_indexer) or (i - 3 in frame_indexer)
                     or (i - 4 in frame_indexer) or (i - 5 in frame_indexer)
                     or (i - 6 in frame_indexer) or (i - 7 in frame_indexer)]
    for i in output_frames:
        half_i = int((frame_i[i] + frame_i[i + 1]) / 2)
        cur_frame_1st_half = novelty_hwr[frame_i[i]: half_i]
        cur_frame_2nd_half = novelty_hwr[half_i: frame_i[i + 1]]
        onset_integrals[2 * i] = np.sum(cur_frame_1st_half)
        onset_integrals[2 * i + 1] = np.sum(cur_frame_2nd_half)

    # Step 2: Calculate the cosine distance between the MFCC values
    for i in frame_indexer:
        onset_correlations[i][0] = np.max(np.correlate(novelty_hwr[frame_i[i - 1]: frame_i[i]],
                                                       novelty_hwr[frame_i[i]: frame_i[i + 1]], mode='valid'))
        onset_correlations[i][1] = np.max(np.correlate(novelty_hwr[frame_i[i]: frame_i[i + 1]],
                                                       novelty_hwr[frame_i[i + 1]: frame_i[i + 2]], mode='valid'))
        onset_correlations[i][2] = np.max(np.correlate(novelty_hwr[frame_i[i]: frame_i[i + 1]],
                                                       novelty_hwr[frame_i[i + 2]: frame_i[i + 3]], mode='valid'))
        onset_correlations[i][3] = np.max(np.correlate(novelty_hwr[frame_i[i]: frame_i[i + 1]],
                                                       novelty_hwr[frame_i[i + 3]: frame_i[i + 4]], mode='valid'))
        onset_correlations[i][4] = onset_integrals[2 * i] - onset_integrals[2 * i - 1]
        onset_correlations[i][5] = onset_integrals[2 * i + 2] + onset_integrals[2 * i + 3] - \
                                   onset_integrals[2 * i - 1] - onset_integrals[2 * i - 2]
        for j in range(1, 16):
            onset_correlations[i][5 + j] = onset_integrals[2 * i + j] - onset_integrals[2 * i]

    # Include the MFCC coefficients as features
    result = onset_correlations[frame_indexer]
    return preprocessing.scale(result)
