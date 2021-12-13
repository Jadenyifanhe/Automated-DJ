import numpy as np
from essentia import *
from essentia.standard import *


class KeyEstimator:

    def __init__(self):
        pass

    def __call__(self, audio):
        FRAME_SIZE = 2048
        HOP_SIZE = FRAME_SIZE // 2
        spec = Spectrum(size=FRAME_SIZE)
        specPeaks = SpectralPeaks()
        hpcp = HPCP()
        key = Key(profileType='edma')
        w = Windowing(type='blackmanharris92')
        pool = Pool()
        for frame in FrameGenerator(audio, frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
            frame_spectrum = spec(w(frame))
            frequencies, magnitudes = specPeaks(frame_spectrum)
            hpcpValue = hpcp(frequencies, magnitudes)
            pool.add('hpcp', hpcpValue)
        hpcp_avg = np.average(pool['hpcp'], axis=0)
        key, scale = key(hpcp_avg)[:2]
        return key, scale
