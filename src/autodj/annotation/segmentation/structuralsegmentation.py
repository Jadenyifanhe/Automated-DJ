import logging

import numpy as np
import scipy.signal
from essentia import *
from essentia.standard import *
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances

essentia.log_active = False
logger = logging.getLogger('colorlogger')


def calculateCheckerboardCorrelation(matrix, N):
    M = min(matrix.shape[0], matrix.shape[1])
    result = np.zeros(M)
    u1 = scipy.signal.gaussian(2 * N, std=N / 2.0).reshape((2 * N, 1))
    u2 = scipy.signal.gaussian(2 * N, std=N / 2.0).reshape((2 * N, 1))
    U = np.dot(u1, np.transpose(u2))
    U[:N, N:] *= -1
    U[N:, :N] *= -1
    matrix_padded = np.pad(matrix, N, mode='edge')
    for index in range(N, N + M):
        submatrix = matrix_padded[index - N:index + N, index - N:index + N]
        result[index - N] = np.sum(submatrix * U)
    return result


def adaptive_mean(x, N):
    return np.convolve(x, [1.0] * int(N), mode='same') / N


class StructuralSegmentator:
    def analyse(self, audio_in, downbeats, onset_curve, tempo):
        pool = Pool()
        w = Windowing(type='hann')
        spectrum = Spectrum()
        mfcc = MFCC()

        first_downbeat_sample = int(44100 * downbeats[0])
        audio = audio_in[first_downbeat_sample:]

        FRAME_SIZE = int(44100 * (60 / tempo) / 2)
        HOP_SIZE = int(FRAME_SIZE / 2)
        for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame[:FRAME_SIZE - (FRAME_SIZE % 2)])))
            pool.add('lowlevel.mfcc', mfcc_coeffs)
            pool.add('lowlevel.mfcc_bands', mfcc_bands)

        selfsim_mfcc = cosine_similarity(np.array(pool['lowlevel.mfcc']), np.array(pool['lowlevel.mfcc']))
        selfsim_mfcc -= np.average(selfsim_mfcc)
        selfsim_mfcc *= (1.0 / np.max(selfsim_mfcc))

        for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            pool.add('lowlevel.rms', np.average(frame ** 2))

        selfsim_rms = pairwise_distances(pool['lowlevel.rms'].reshape(-1, 1))
        selfsim_rms -= np.average(selfsim_rms)
        selfsim_rms *= (1.0 / np.max(selfsim_rms))

        novelty_mfcc = calculateCheckerboardCorrelation(selfsim_mfcc, N=32)
        novelty_mfcc *= 1.0 / np.max(novelty_mfcc)

        novelty_rms = np.abs(calculateCheckerboardCorrelation(selfsim_rms, N=32))
        novelty_rms *= 1.0 / np.max(np.abs(novelty_rms))

        novelty_product = novelty_rms * novelty_mfcc
        novelty_product = [i if i > 0 else 0 for i in novelty_product]
        novelty_product = np.sqrt(novelty_product)

        peaks_absmax_i = np.argmax(novelty_product)
        peaks_absmax = novelty_product[peaks_absmax_i]
        threshold = peaks_absmax * 0.05
        peakDetection = PeakDetection(interpolate=False, maxPeaks=100, orderBy='amplitude', range=len(novelty_product),
                                      maxPosition=len(novelty_product), threshold=threshold)
        peaks_pos, peaks_ampl = peakDetection(novelty_product.astype('single'))
        peaks_ampl = peaks_ampl[np.argsort(peaks_pos)]
        peaks_pos = peaks_pos[np.argsort(peaks_pos)]

        peaks_pos_modified, peaks_ampl_modified = [], []
        peaks_pos_dbindex = []

        peak_idx = 0
        peak_cur_s = (HOP_SIZE / 44100) * peaks_pos[peak_idx]
        num_filtered_out = 0

        downbeat_len_s = 4 * 60 / tempo
        delta = 0.4
        for dbindex, downbeat in zip(range(len(downbeats)), np.array(downbeats) - downbeats[0]):
            while peak_cur_s < downbeat - delta * downbeat_len_s and peak_idx < len(peaks_pos):
                num_filtered_out += 1
                peak_idx += 1
                if peak_idx != len(peaks_pos):
                    peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
            if peak_idx == len(peaks_pos):
                break
            while peak_cur_s < downbeat + delta * downbeat_len_s and peak_idx < len(peaks_pos):
                peak_newpos = int(downbeat * 44100.0 / HOP_SIZE)
                peaks_pos_modified.append(peak_newpos)
                peaks_ampl_modified.append(peaks_ampl[peak_idx])
                peaks_pos_dbindex.append(dbindex)
                peak_idx += 1
                if peak_idx != len(peaks_pos):
                    peak_cur_s = (HOP_SIZE / 44100.0) * peaks_pos[peak_idx]
            if peak_idx == len(peaks_pos):
                break

        peaks_ampl_modified = np.array(peaks_ampl_modified)
        peaks_pos_dbindex = np.array(peaks_pos_dbindex)
        NUM_HIGHEST_PEAKS = 20
        highest_peaks_db_indices = (peaks_pos_dbindex[np.argsort(peaks_ampl_modified)])[-NUM_HIGHEST_PEAKS:]
        highest_peaks_amplitudes = (peaks_ampl_modified[np.argsort(peaks_ampl_modified)])[-NUM_HIGHEST_PEAKS:]
        distances8 = []
        distances8_high = []

        for i in range(8):
            distances8.append(sum([h for p, h in zip(highest_peaks_db_indices, highest_peaks_amplitudes)
                                   if (p - i) % 8 == 0 and (p + 1 not in highest_peaks_db_indices or
                                                            np.max(highest_peaks_amplitudes[
                                                                       highest_peaks_db_indices == p + 1]) < 0.75 * h)
                                   and (p + 2 not in highest_peaks_db_indices or
                                        np.max(
                                            highest_peaks_amplitudes[highest_peaks_db_indices == p + 2]) < 0.75 * h)]))
            distances8_high.append(len([p for p in highest_peaks_db_indices if (p - i) % 8 == 0]))

        most_likely_8db_index = np.argmax(distances8 * np.array(distances8_high).astype(float))
        last_downbeat = len(downbeats) - 1

        segment_indices = [most_likely_8db_index if most_likely_8db_index <= 4 else most_likely_8db_index - 8]
        last_boundary = last_downbeat - ((last_downbeat - most_likely_8db_index) % 8)
        segment_indices.extend([last_boundary])
        segment_indices.extend([db for db in highest_peaks_db_indices if (db - most_likely_8db_index) % 8 == 0])
        segment_indices.extend([db + 1 for db in highest_peaks_db_indices if (db + 1 - most_likely_8db_index) % 8 == 0])
        segment_indices.extend([db + 2 for db in highest_peaks_db_indices if (db + 2 - most_likely_8db_index) % 8 == 0])
        segment_indices = np.unique(sorted(segment_indices))

        adaptive_mean_rms = adaptive_mean(pool['lowlevel.rms'], 64)
        mean_rms = np.mean(adaptive_mean_rms)
        adaptive_mean_odf = adaptive_mean(onset_curve, int((44100 * 60 / tempo) / 512) * 4)
        mean_odf = np.mean(adaptive_mean_odf)

        segment_types = []

        def getSegmentType(dbindex):
            if dbindex >= last_boundary:
                return 'L'
            after_index = int(int((dbindex + 4) * 4 * 60.0 / tempo * 44100.0) / HOP_SIZE)
            rms_after = adaptive_mean_rms[after_index] / mean_rms
            after_index = int(int((dbindex + 4) * 4 * 60.0 / tempo * 44100.0) / 512)
            odf_after = adaptive_mean_odf[after_index] / mean_odf
            return 'H' if rms_after >= 1.0 and odf_after >= 1.0 else 'L'

        for segment in segment_indices:
            segment_types.append(getSegmentType(segment))

        additional_segment_indices = []
        additional_segment_types = []
        for i in range(len(segment_indices) - 1):
            if segment_indices[i + 1] - segment_indices[i] >= 32:
                previous_type = segment_types[i]
                for offset in range(16, segment_indices[i + 1] - segment_indices[i], 16):
                    if getSegmentType(segment_indices[i] + offset) != previous_type:
                        additional_segment_indices.append(segment_indices[i] + offset)
                        previous_type = 'H' if previous_type == 'L' else 'L'
                        additional_segment_types.append(previous_type)

        segment_indices = np.append(segment_indices, additional_segment_indices)
        segment_types = np.append(segment_types, additional_segment_types)
        permutation = np.argsort(segment_indices)
        segment_indices = segment_indices[permutation].astype('int')
        segment_types = segment_types[permutation]

        return segment_indices, segment_types
