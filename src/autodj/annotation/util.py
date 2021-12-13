import csv
import logging
import os

import numpy as np
from essentia import *
from essentia.standard import *
from essentia.standard import AudioOnsetsMarker

ANNOT_SUBDIR = '_annot_auto/'
ANNOT_DOWNB_PREFIX = 'downbeats_'
ANNOT_BEATS_PREFIX = 'beats_'
ANNOT_SEGMENT_PREFIX = 'segments_'
ANNOT_GAIN_PREFIX = 'gain_'
ANNOT_KEY_PREFIX = 'key_'
ANNOT_SPECTRALCONTRAST_PREFIX = 'specctrst_'
ANNOT_THEME_DESCR_PREFIX = 'theme_descr_'
ANNOT_SINGINGVOICE_PREFIX = 'singing_'
ANNOT_ODF_HFC_PREFIX = 'odf_hfc_'
ANNOT_MARKED_PREFIX = '_fix_annotations_'
logger = logging.getLogger('colorlogger')


class UnannotatedException(Exception):
    pass


def pathAnnotationFile(directory, song_title, prefix):
    return os.path.join(directory, ANNOT_SUBDIR, prefix + song_title + '.txt')


def loadCsvAnnotationFile(directory, prefix):
    result = {}
    try:
        with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'r+') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            for row in reader:
                key, value = row
                try:
                    value = float(value)
                except ValueError:
                    pass
                result[key] = value
    except IOError as e:
        logger.debug('Csv annotation file not found, silently ignoring exception ' + str(e))
    return result


def writeCsvAnnotation(directory, prefix, song_title, value):
    with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'a+') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        if type(value) is float:
            writer.writerow([song_title, '{:.9f}'.format(value)])
        else:
            writer.writerow([song_title, '{:}'.format(value)])


def deleteCsvAnnotation(directory, prefix, song_title):
    titles = []
    with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'r+') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for line in reader:
            titles.append(line)
    with open(os.path.join(directory, ANNOT_SUBDIR, prefix + '.csv'), 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for line in titles:
            if line[0] != song_title:
                writer.writerow(line)


def loadAnnotationFile(directory, song_title, prefix):
    """
    Loads an input file with annotated times in seconds.
    """
    input_file = pathAnnotationFile(directory, song_title, prefix)
    result = []
    result_dict = {}
    if os.path.exists(input_file):
        with open(input_file) as f:
            for line in f:
                if line[0] == '#':
                    try:
                        key, value = str.split(line[1:], ' ')
                        result_dict[key] = float(value)
                    except ValueError:
                        pass
                else:
                    result.append(line)
    else:
        raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
    return result, result_dict


def writeAnnotFile(directory, song_title, prefix, array, values_dict=None):
    if values_dict is None:
        values_dict = {}
    output_file = pathAnnotationFile(directory, song_title, prefix)
    with open(output_file, 'w+') as f:
        for key, value in values_dict.items():
            f.write('#' + str(key) + ' ' + '{:.9f}'.format(value) + '\n')
        for value in array:
            if type(value) is tuple:
                for v in value:
                    if type(v) is float:
                        f.write('{:.9f} '.format(v))
                    else:
                        f.write('{} '.format(v))
                f.write('\n')
            else:
                f.write("{:.9f}".format(value) + '\n')


def loadBinaryAnnotFile(directory, song_title, prefix):
    input_file = pathAnnotationFile(directory, song_title, prefix)
    if os.path.exists(input_file):
        result = np.load(input_file)
    else:
        raise UnannotatedException('Attempting to load annotations of unannotated audio' + input_file + '!')
    return result


def writeBinaryAnnotFile(directory, song_title, prefix, array):
    output_file = pathAnnotationFile(directory, song_title, prefix)
    np.save(output_file, array)


def overlayAudio(audio, beats):
    onsetMarker = AudioOnsetsMarker(onsets=1.0 * beats)
    audioMarked = onsetMarker(audio)
    return audioMarked
