import os

import essentia
import essentia.standard as ess
import numpy as np
from sklearn.externals import joblib

basepath = os.path.dirname(os.path.abspath(__file__))


class ThemeDescriptorEstimator:

    def __init__(self, ):
        basepath = os.path.dirname(os.path.abspath(__file__))
        self.theme_scaler = joblib.load(os.path.join(basepath, 'song_theme_scaler_2.pkl'))
        self.theme_pca = joblib.load(os.path.join(basepath, 'song_theme_pca_model_python3.pkl'))

    def __call__(self, audio, slices):
        FRAME_SIZE = 2048
        HOP_SIZE = FRAME_SIZE // 2

        spec = ess.Spectrum(size=FRAME_SIZE)
        w = ess.Windowing(type='hann')
        pool = essentia.Pool()
        specContrast = ess.SpectralContrast(frameSize=FRAME_SIZE, sampleRate=44100, numberBands=12)

        for start_sample, end_sample in slices:
            for frame in ess.FrameGenerator(audio[start_sample:end_sample], frameSize=FRAME_SIZE, hopSize=HOP_SIZE):
                frame_spectrum = spec(w(frame))
                specCtrst, specValley = specContrast(frame_spectrum)
                pool.add('audio.spectralContrast', specCtrst)
                pool.add('audio.spectralValley', specValley)

        def calculateDeltas(array):
            return array[1:] - array[:-1]

        specCtrstAvgs = np.average(pool['audio.spectralContrast'], axis=0)
        specValleyAvgs = np.average(pool['audio.spectralValley'], axis=0)
        specCtrstDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralContrast'])), axis=0)
        specValleyDeltas = np.average(np.abs(calculateDeltas(pool['audio.spectralValley'])), axis=0)
        features = np.concatenate((specCtrstAvgs, specValleyAvgs, specCtrstDeltas, specValleyDeltas))

        result = self.theme_pca.transform(
            self.theme_scaler.transform(features.astype('single').reshape((1, -1)))).astype('single')

        return result
