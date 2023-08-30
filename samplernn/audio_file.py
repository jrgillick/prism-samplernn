import warnings
import librosa
import soundfile as sf
import numpy as np
import random

# Audio augmentation code adapted from https://github.com/iver56/audiomentations
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Reverse, SevenBandParametricEQ

augment_audio = Compose([
    Shift(min_shift=-0.05, max_shift=0.05, shift_unit="fraction", fade_duration=0.01, p=0.9),
    AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.0002, p=0.1),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Reverse(p=0.1),
    SevenBandParametricEQ(min_gain_db=-3, max_gain_db=3, p=0.5)
])

# Contains some code adapted from WaveNet
# https://github.com/ibab/tensorflow-wavenet/blob/master/wavenet/audio_reader.py

# warnings.simplefilter("always")

def randomize(list):
    list_idx = [i for i in range(len(list))]
    random.shuffle(list_idx)
    for idx in range(len(list)):
        yield list[list_idx[idx]]

def yield_from_list(list, shuffle=True):
    list_idx = [i for i in range(len(list))]
    if shuffle==True : random.shuffle(list_idx)
    for idx in range(len(list)):
        yield list[list_idx[idx]]







#def load_audio(files, shuffle=True, augment=False, sr=None):
#    '''Generator that yields audio waveforms from the directory.'''
#    print('Corpus length: {} files.'.format(len(files)))
#    for filename in yield_from_list(files, shuffle=shuffle):
#        (audio, _) = librosa.load(filename, sr=sr, mono=True)
#        audio = audio.reshape(-1, 1)
#        print("Loading corpus entry {}".format(filename))
#        yield audio
def load_audio(files, shuffle=True, augment=False, sr=None):
    '''Generator that yields audio waveforms from the directory.'''
    print('Corpus length: {} files.'.format(len(files)))
    for filename in yield_from_list(files, shuffle=shuffle):
        (audio, sr) = librosa.load(filename, sr=sr, mono=True)
        if augment:
          augmented_audio = augment_audio(samples=audio, sample_rate=sr)
          print("Loading augmented corpus entry {}".format(filename))
          augmented_audio = augmented_audio.reshape(-1, 1)
          yield augmented_audio
        else:
          print("Loading corpus entry {}".format(filename))
          audio = audio.reshape(-1, 1)
          yield audio

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)