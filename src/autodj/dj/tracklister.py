import random

from scipy.spatial.distance import euclidean as euclidean_distance

from . import songcollection
from . import songtransitions
from .song import *

logger = logging.getLogger('colorlogger')

TYPE_DOUBLE_DROP = 'double drop'
TYPE_ROLLING = 'rolling'
TYPE_CHILL = 'relaxed'

TRANSITION_PROBAS = {
    TYPE_CHILL: [0.0, 0.7, 0.3],
    TYPE_ROLLING: [0.2, 0.8, 0.0],
    TYPE_DOUBLE_DROP: [0.2, 0.8, 0.0]
}

LENGTH_ROLLING_IN = 16
LENGTH_ROLLING_OUT = 16
LENGTH_DOUBLE_DROP_IN = LENGTH_ROLLING_IN
LENGTH_DOUBLE_DROP_OUT = 32
LENGTH_CHILL_IN = 16
LENGTH_CHILL_OUT = 16

THEME_WEIGHT = 0.4
PREV_SONG_WEIGHT = - 0.1 * (1 - THEME_WEIGHT)
CURRENT_SONG_WEIGHT = 1 - (THEME_WEIGHT + PREV_SONG_WEIGHT)

NUM_SONGS_IN_KEY_MINIMUM = 5
NUM_SONGS_ONSETS = 3
MAX_SONGS_IN_SAME_KEY = 3

ROLLING_START_OFFSET = LENGTH_ROLLING_IN + LENGTH_ROLLING_OUT


def is_vocal_clash_pred(master, slave):
    master = 2 * master[1:-1] + master[:-2] + master[2:] >= 2
    slave = 2 * slave[1:-1] + slave[:-2] + slave[2:] >= 2
    return sum(np.logical_and(master, slave)) >= 2


def getDbeatAfter(dbeat, options, n=1):
    if dbeat is None:
        return None
    candidates = [b for b in options if b > dbeat]
    if len(candidates) < n:
        return None
    else:
        return candidates[n - 1]


def getHAfter(song, dbeat, n=1):
    options = [song.segment_indices[i] for i in range(len(song.segment_indices)) if song.segment_types[i] == 'H']
    return getDbeatAfter(dbeat, options, n=n)


def getLAfter(song, dbeat, n=1):
    options = [song.segment_indices[i] for i in range(len(song.segment_indices)) if song.segment_types[i] == 'L']
    return getDbeatAfter(dbeat, options, n=n)


def getAllMasterSwitchPoints(song, fade_type):
    types, indices = song.segment_types, song.segment_indices
    LH = [indices[i] for i in range(1, len(indices)) if types[i - 1] == 'L' and types[i] == 'H']
    HL = [indices[i] for i in range(1, len(indices)) if types[i - 1] == 'H' and types[i] == 'L']

    if fade_type == TYPE_DOUBLE_DROP:
        cues = [i - 1 for i in LH if i <= indices[-1] - LENGTH_DOUBLE_DROP_OUT]
        L_fade_in = [min(LENGTH_DOUBLE_DROP_IN, i - indices[0]) - 1 for i in cues]
        L_fade_out = [LENGTH_DOUBLE_DROP_OUT + 1 for i in cues]

    elif fade_type == TYPE_ROLLING:
        cues = [i - LENGTH_ROLLING_OUT - LENGTH_ROLLING_IN - 1 for i in HL if
                i - LENGTH_ROLLING_OUT > 0 and i <= indices[-1] - LENGTH_ROLLING_OUT]
        L_fade_in = [min(LENGTH_ROLLING_IN, i - indices[0]) - 1 for i in cues]
        L_fade_out = [min(LENGTH_ROLLING_OUT, getLAfter(song, i) - i) + 1 for i in cues]

    elif fade_type == TYPE_CHILL:
        cues = HL
        L_fade_in = [min(LENGTH_CHILL_IN, i - indices[0]) for i in cues]
        L_fade_out = [
            min(LENGTH_CHILL_OUT, (getHAfter(song, i) if not (getHAfter(song, i) is None) else indices[-1]) - i) for i
            in cues]

    else:
        raise Exception('Unknown fade type {}'.format(fade_type))

    return list(zip(cues, L_fade_in, L_fade_out))


def getAllSlaveCues(song, fade_type, min_playable_length=32):
    types, indices = song.segment_types, song.segment_indices

    LH = [indices[i] for i in range(1, len(indices)) if
          types[i - 1] == 'L' and types[i] == 'H' and indices[-1] - indices[i] > min_playable_length]

    if fade_type == TYPE_DOUBLE_DROP:
        cues = [i - 1 for i in LH]
        fade_in_lengths = [min(i, LENGTH_DOUBLE_DROP_IN - 1) for i in cues]
    elif fade_type == TYPE_ROLLING:
        cues = [i - 1 for i in LH]
        fade_in_lengths = [min(i, LENGTH_ROLLING_IN - 1) for i in cues]
    elif fade_type == TYPE_CHILL:
        cues = [indices[0] + LENGTH_CHILL_IN]
        fade_in_lengths = [LENGTH_CHILL_IN]
    else:
        raise Exception('Unknown fade type {}'.format(fade_type))

    return list(zip(cues, fade_in_lengths))


def getMasterQueue(song, start_dbeat, cur_fade_type):
    start_dbeat = start_dbeat + (8 - (start_dbeat - song.segment_indices[0]) % 8) % 8

    P_chill, P_roll, P_ddrop = TRANSITION_PROBAS[cur_fade_type]

    if P_ddrop > 0:

        isDoubleDrop = (random.random() <= P_ddrop)
        cues = getAllMasterSwitchPoints(song, TYPE_DOUBLE_DROP)
        cues = [c for c in cues if c[0] >= start_dbeat]

        if isDoubleDrop and len(cues) != 0:
            doubleDropDbeat, max_fade_in_len, fade_out_len = cues[0]
            max_fade_in_len = min(max_fade_in_len, doubleDropDbeat - start_dbeat - 1)
            return doubleDropDbeat - max_fade_in_len, TYPE_DOUBLE_DROP, max_fade_in_len, fade_out_len

    P_roll = P_roll / (P_roll + P_chill)

    if P_roll > 0:

        isRolling = (random.random() <= P_roll)
        cues = getAllMasterSwitchPoints(song, TYPE_ROLLING)
        cues = [c for c in cues if c[0] >= start_dbeat + ROLLING_START_OFFSET - 1]

        if isRolling and len(cues) != 0:
            rollingDbeat, max_fade_in_len, fade_out_len = cues[0]
            max_fade_in_len = min(max_fade_in_len, rollingDbeat - start_dbeat - 1)
            return rollingDbeat - max_fade_in_len, TYPE_ROLLING, max_fade_in_len, fade_out_len

    cues = getAllMasterSwitchPoints(song, TYPE_CHILL)
    cues = [c for c in cues if c[0] >= start_dbeat]
    if len(cues) == 0:
        cue = start_dbeat
        max_fade_in_len = 0
        fade_out_len = min(LENGTH_CHILL_OUT, song.segment_indices[-1] - start_dbeat)
    else:
        cue, max_fade_in_len, fade_out_len = cues[0]
    max_fade_in_len = min(max_fade_in_len, cue - start_dbeat)
    return cue - max_fade_in_len, TYPE_CHILL, max_fade_in_len, fade_out_len


def getSlaveQueue(song, fade_type, min_playable_length=32):
    cues = getAllSlaveCues(song, fade_type, min_playable_length)

    if fade_type == TYPE_DOUBLE_DROP or fade_type == TYPE_ROLLING:
        if len(cues) > 0:
            cue, fade_in_len = cues[np.random.randint(len(cues))]
            return cue, fade_in_len
        else:
            cues = getAllSlaveCues(song, TYPE_CHILL, min_playable_length)
            logger.debug('Warning: no H dbeats!')

    cue, fade_in_len = cues[0]
    return cue, fade_in_len


def calculateOnsetSimilarity(odf1, odf2):
    if len(odf1) < len(odf2):
        temp = odf1
        odf1 = odf2
        odf2 = temp
    avg1 = np.average(odf1)
    if avg1 != 0:
        odf1 /= avg1
    avg2 = np.average(odf2)
    if avg2 != 0:
        odf2 /= avg2

    N = 2
    scores = [0] * (2 * N + 1)
    prev_scores = scores
    prev_i2_center = 0
    slope = float(len(odf2)) / len(odf1)

    for i1 in range(len(odf1)):
        i2_center = int(i1 * slope + 0.5)
        for i in range(0, 2 * N + 1):
            i2 = i2_center - N + i
            if i2 >= len(odf2) or i2 < 0:
                break
            score_increment = abs(odf1[i1] - odf2[i2])

            if prev_i2_center == i2_center:
                score_new = prev_scores[i]
                if i > 0:
                    score_new = min(score_new, scores[i - 1])
                    score_new = min(score_new, prev_scores[i - 1])
            else:
                score_new = prev_scores[i]
                if i < 2 * N:
                    score_new = min(score_new, prev_scores[i + 1])
                if i > 0:
                    score_new = min(score_new, scores[i - 1])
            scores[i] = score_new + score_increment
        prev_i2_center = i2_center
        prev_scores = scores
    return scores[N]


class TrackLister:
    def __init__(self, song_collection):
        self.songs = None
        self.crossfades = None
        self.song_collection = song_collection
        self.songsUnplayed = song_collection.get_annotated()
        self.songsPlayed = []
        self.song_file_idx = 0
        self.semitone_offset = 0

        self.theme_centroid = None
        self.prev_song_theme_descriptor = None

    def getFirstSong(self):
        self.songsUnplayed = self.song_collection.get_annotated()
        firstSong = np.random.choice(self.songsUnplayed, size=1)[0]
        self.songsUnplayed.remove(firstSong)
        self.songsPlayed.append(firstSong)
        firstSong.open()

        self.chooseNewTheme(firstSong)
        self.prev_song_theme_descriptor = firstSong.song_theme_descriptor

        return firstSong

    def chooseNewTheme(self, firstSong):
        songs_distance_to_first_song = []
        songs_themes = []
        for song in self.songsUnplayed:
            song.open()
            theme = song.song_theme_descriptor
            songs_themes.append(firstSong.song_theme_descriptor)
            songs_distance_to_first_song.append(euclidean_distance(theme, firstSong.song_theme_descriptor))
            song.close()
        songs_sorted = np.argsort(songs_distance_to_first_song)
        indices_sorted = songs_sorted[:int(len(songs_sorted) / 4)]
        self.theme_centroid = np.average(np.array(songs_themes)[indices_sorted], axis=0)

    def getSongOptionsInKey(self, key, scale):
        songs_in_key = []
        songs_unplayed = self.songsUnplayed
        keys_added = []

        def addSongsInKey(key, scale):
            if (key, scale) not in keys_added:
                titles = self.song_collection.get_titles_in_key(key, scale)
                songs_to_add = [s for s in songs_unplayed if s.title in titles]
                songs_in_key.extend(songs_to_add)
                keys_added.append((key, scale))

        closely_related_keys = songcollection.get_closely_related_keys(key, scale)
        for key_, scale_ in closely_related_keys:
            addSongsInKey(key_, scale_)
            key_to_add, scale_to_add = songcollection.get_key_transposed(key_, scale_, -1)
            addSongsInKey(key_to_add, scale_to_add)

        if len(songs_in_key) == 0:
            logger.debug('Not enough songs in pool, adding all songs!')
            songs_in_key = self.songsUnplayed
        return np.array(songs_in_key)

    def filterSongOptionsByThemeDistance(self, song_options, master_song):
        song_options_distance_to_centroid = []
        cur_theme_centroid = THEME_WEIGHT * self.theme_centroid \
                             + CURRENT_SONG_WEIGHT * master_song.song_theme_descriptor \
                             + PREV_SONG_WEIGHT * self.prev_song_theme_descriptor
        for song in song_options:
            song.open()
            dist_to_centroid = euclidean_distance(cur_theme_centroid, song.song_theme_descriptor)
            song_options_distance_to_centroid.append(dist_to_centroid)
            song.close()
        song_options_closest_to_centroid = np.argsort(song_options_distance_to_centroid)

        for i in song_options_closest_to_centroid[:NUM_SONGS_ONSETS]:
            song_options[i].open()
            song_options[i].close()

        return song_options[song_options_closest_to_centroid[:NUM_SONGS_ONSETS]]

    def getBestNextSongAndCrossfade(self, master_song, master_cue, master_fade_in_len, fade_out_len, fade_type):
        transition_length = master_fade_in_len + fade_out_len

        key, scale = songcollection.get_key_transposed(master_song.key, master_song.scale, self.semitone_offset)
        song_options = self.getSongOptionsInKey(key, scale)
        closely_related_keys = songcollection.get_closely_related_keys(key, scale)

        song_options = self.filterSongOptionsByThemeDistance(song_options, master_song)

        master_song.open()
        best_score = np.inf
        best_score_clash = np.inf
        best_song = None
        best_fade_in_len = None
        best_slave_cue = None
        best_master_cue = None
        best_song_clash = None
        best_fade_in_len_clash = None
        best_slave_cue_clash = None
        best_master_cue_clash = None

        for s in song_options:
            next_song = s
            next_song.open()
            queue_slave, fade_in_len = getSlaveQueue(next_song, fade_type, min_playable_length=transition_length + 16)
            fade_in_len = min(fade_in_len, master_fade_in_len)
            fade_in_len_correction = master_fade_in_len - fade_in_len
            master_cue_corr = master_cue + fade_in_len_correction
            transition_len_corr = transition_length - fade_in_len_correction
            queue_slave = queue_slave - fade_in_len

            if queue_slave >= 16:
                cf = songtransitions.CrossFade(0, [queue_slave], transition_len_corr, fade_in_len, fade_type)
            else:
                cf = songtransitions.CrossFade(0, [queue_slave], transition_len_corr, fade_in_len, fade_type)

            for queue_slave_cur in cf.queue_2_options:
                odf_segment_len = 4
                odf_scores = []
                for odf_start_dbeat in range(0, transition_len_corr, odf_segment_len):
                    odf_master = master_song.getOnsetCurveFragment(
                        master_cue_corr + odf_start_dbeat,
                        min(master_cue_corr + odf_start_dbeat + odf_segment_len, master_cue_corr + transition_len_corr))
                    odf_slave = s.getOnsetCurveFragment(
                        queue_slave_cur + odf_start_dbeat,
                        min(queue_slave_cur + odf_start_dbeat + odf_segment_len, queue_slave_cur + transition_len_corr))
                    onset_similarity = calculateOnsetSimilarity(odf_master, odf_slave) / odf_segment_len
                    odf_scores.append(onset_similarity)

                singing_master = np.array(
                    master_song.singing_voice[master_cue_corr: master_cue_corr + transition_len_corr] > 0)
                singing_slave = np.array(s.singing_voice[queue_slave: queue_slave + transition_len_corr] > 0)
                singing_clash = is_vocal_clash_pred(singing_master, singing_slave)

                onset_similarity = np.average(odf_scores)
                score = onset_similarity

                if score < best_score and not singing_clash:
                    best_song = next_song
                    best_score = score
                    best_fade_in_len = fade_in_len
                    best_slave_cue = queue_slave_cur
                    best_master_cue = master_cue_corr
                elif best_score == np.inf and score < best_score_clash and singing_clash:
                    best_song_clash = next_song
                    best_score_clash = score
                    best_fade_in_len_clash = fade_in_len
                    best_slave_cue_clash = queue_slave_cur
                    best_master_cue_clash = master_cue_corr

        if best_song is None:
            best_song = best_song_clash
            best_fade_in_len = best_fade_in_len_clash
            best_slave_cue = best_slave_cue_clash
            best_master_cue = best_master_cue_clash

        if (best_song.key, best_song.scale) not in closely_related_keys:
            shifted_key_up, shifted_scale_up = songcollection.get_key_transposed(best_song.key, best_song.scale, 1)
            if (shifted_key_up, shifted_scale_up) in closely_related_keys:
                self.semitone_offset = 1
            else:
                self.semitone_offset = -1
            logger.debug(
                'Pitch shifting! {} {} by {} semitones'.format(best_song.key, best_song.scale, self.semitone_offset))
        else:
            self.semitone_offset = 0

        self.prev_song_theme_descriptor = master_song.song_theme_descriptor
        self.songsPlayed.append(best_song)
        self.songsUnplayed.remove(best_song)
        if len(self.songsUnplayed) <= NUM_SONGS_IN_KEY_MINIMUM:
            logger.debug('Replenishing song pool')
            self.songsUnplayed.extend(self.songsPlayed)
            self.songsPlayed = []

        return best_song, best_slave_cue, best_master_cue, best_fade_in_len, self.semitone_offset
