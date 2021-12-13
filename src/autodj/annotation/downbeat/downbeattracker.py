import os

import numpy as np
from sklearn.externals import joblib

from . import features
from .features import loudness, mfcc, onsetflux, onsetcsd, onsethfc

feature_modules = [features.loudness, features.mfcc, features.onsetflux, features.onsetcsd, features.onsethfc]


class DownbeatTracker:
    """
    Detects the downbeat locations given the beat locations and audio
    """

    def __init__(self):
        self.model = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'model.pkl'))
        self.scaler = joblib.load(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'scaler.pkl'))

    def trimAudio(self, audio, beats):
        beats = np.array(beats) * 44100  # Beats in samples
        rms = []
        for i in range(len(beats) - 1):
            rms.append(np.sqrt(np.mean(np.square(audio[int(beats[i]): int(beats[i + 1])]))))

        def adaptive_mean(x, N):
            return np.convolve(x, [1.0] * int(N), mode='same') / N

        rms_adaptive = adaptive_mean(rms, 4)
        rms_adaptive_max = max(rms_adaptive)
        start, end, ratiox = 0, 0, 0
        ratios = [.9, .8, .7, .6, .5, .4, .3, .2, .1]
        for ratio in ratios:
            for i in range(len(rms)):
                if rms[i] > ratio * rms_adaptive_max:
                    start = i
                    break
            for i in range(len(rms)):
                if rms[len(rms) - i - 1] > ratio * rms_adaptive_max:
                    end = len(rms) - i - 1
                    break
        return start, end

    def getFeaturesForAudio(self, input_features):
        FRAME_INDEXER_MIN = 4
        FRAME_INDEXER_MAX = len(input_features['beats']) - 9
        trim_start_beat, trim_end_beat = self.trimAudio(input_features['audio'], input_features['beats'])
        indexer_start = np.max(FRAME_INDEXER_MIN, trim_start_beat)
        indexer_end = np.min(FRAME_INDEXER_MAX, trim_end_beat)
        frame_indexer = range(indexer_start, indexer_end)
        features_cur_file = None
        for module in feature_modules:
            absolute_feature_submatrix = module.feature_allframes(input_features, frame_indexer)
            if features_cur_file is None:
                features_cur_file = absolute_feature_submatrix
            else:
                features_cur_file = np.append(features_cur_file, absolute_feature_submatrix, axis=1)
        return features_cur_file, trim_start_beat

    def track(self, audio, beats, fft_mag, fft_phase, onset_curve):
        """
        Track the downbeats of the given audio file
        """
        input_features = {
            'audio': audio,
            'beats': beats,
            'fft_mag': fft_mag,
            'fft_ang': fft_phase,
            'onset_curve': onset_curve,
        }
        features, trim_start_beat = self.getFeaturesForAudio(input_features)
        probas = self.model.predict_log_proba(features)
        sum_log_probas = np.array([[0, 0, 0, 0]], dtype='float64')
        permuted_row = [0] * 4
        for i, j, row in zip(range(len(probas)), np.array(range(len(probas))) % 4, probas):
            permuted_row[:4 - j] = row[j:]
            permuted_row[4 - j:] = row[:j]
            sum_log_probas = sum_log_probas + permuted_row
        downbeatIndex = ((4 - np.argmax(sum_log_probas)) + trim_start_beat) % 4
        return beats[downbeatIndex::4]
