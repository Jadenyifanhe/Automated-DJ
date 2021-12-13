import librosa.core as core
import librosa.decompose as decompose
import librosa.util
import numpy as np
from scipy import signal, interpolate


def crossfade(audio1, audio2, length=None):
    if length is None:
        length = min(audio1.size, audio2.size)
    profile = ((np.arange(0.0, length)) / length)
    output = (audio1[:length] * profile[::-1]) + (audio2[:length] * profile)
    return output[:length]


def time_stretch_hpss(audio, f):
    if f == 1:
        return audio

    stft = core.stft(audio)

    stft_harm, stft_perc = decompose.hpss(stft, kernel_size=31)  # original kernel size 31

    y_perc = librosa.util.fix_length(core.istft(stft_perc, dtype=audio.dtype), len(audio))
    y_perc = time_stretch_sola(y_perc, f, wsola=True)

    stft_stretch = core.phase_vocoder(stft_harm, 1 / f)
    y_harm = librosa.util.fix_length(core.istft(stft_stretch, dtype=y_perc.dtype), len(y_perc))

    return y_harm + y_perc


def time_stretch_sola(audio, f, wsola=False):
    if f == 1:
        return audio

    frame_len_1 = 4096 if wsola else 1024
    overlap_len = frame_len_1 / 8
    frame_len_0 = frame_len_1 - overlap_len
    next_frame_offset_f = frame_len_1 / f
    seek_win_len_half = frame_len_1 / 16

    def find_matching_frame(frame, theor_center):
        cur_win_min = theor_center - seek_win_len_half
        cur_win_max = theor_center + seek_win_len_half
        correlation = signal.fftconvolve(
            audio[int(cur_win_min):int(cur_win_max + len(frame))], frame[::-1], mode='valid')
        optimum = np.argmax(correlation[:int(2 * seek_win_len_half)])

        return theor_center + (optimum - seek_win_len_half)

    num_samples_out = int(f * audio.size)
    output = np.zeros(num_samples_out + int(frame_len_1))

    num_frames_out = num_samples_out / frame_len_1
    in_ptr_th_f = 0.0
    in_ptr = 0

    for out_ptr in range(0, int(num_frames_out * frame_len_1), int(frame_len_1)):
        frame_to_copy = audio[int(in_ptr): int(in_ptr + frame_len_0)]
        output[out_ptr: out_ptr + len(frame_to_copy)] = frame_to_copy
        if in_ptr + frame_len_1 > audio.size:
            frame_to_copy = audio[int(in_ptr + frame_len_0): int(in_ptr + frame_len_1)]
            output[int(out_ptr + frame_len_0): int(out_ptr + frame_len_0 + len(frame_to_copy))] = frame_to_copy
            return output

        frame_to_match = audio[int(in_ptr + frame_len_0): int(in_ptr + frame_len_0 + frame_len_1)]
        if wsola:
            match_ptr = find_matching_frame(frame_to_match, int(in_ptr_th_f + next_frame_offset_f) - overlap_len)
        else:
            match_ptr = int(in_ptr_th_f + next_frame_offset_f) - overlap_len

        frame1_overlap = audio[int(in_ptr + frame_len_0): int(in_ptr + frame_len_1 + 1)]
        frame2_overlap = audio[int(match_ptr): int(match_ptr + overlap_len + 1)]

        temp = crossfade(frame1_overlap, frame2_overlap)
        output[int(out_ptr + frame_len_0): int(out_ptr + frame_len_0 + len(temp))] = temp

        in_ptr = match_ptr + overlap_len
        in_ptr_th_f += next_frame_offset_f

    return np.array(output).astype('single')


def time_stretch_and_pitch_shift(audio, f, semitones=0):
    semitone_factor = np.power(2.0, semitones / 12.0)

    audio = time_stretch_hpss(audio, f * semitone_factor)

    if semitones != 0:
        x = range(audio.size)
        x_new = np.linspace(0, audio.size - 1, int(audio.size / semitone_factor))
        f = interpolate.interp1d(x, audio, kind='quadratic')
        audio = f(x_new)
    return audio
