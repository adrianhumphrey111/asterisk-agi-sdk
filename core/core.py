#!/usr/bin/vai-agi-python-path
import numpy as np
import dialplan
import librosa
import soundfile as sf

from scikits.audiolab import Format, Sndfile
from tempfile import mkstemp
from constants import constants
from fsm_interface.hermes import Hermes
from utils.read_stdin import get_stdn_var, exit_AGI
from dialplan import stdin
from utils import helper_functions as fn
from utils.log import __console
from utils import sys_vars


host_url        = sys_vars.host_url
session_id      = sys_vars.session_id
client_id       = sys_vars.client_id
access_token    = sys_vars.access_token
virtual_number  = sys_vars.virtual_number
caller_id       = sys_vars.caller_id


hermes = Hermes(
    host_url,
    session_id,
    client_id,
    access_token,
    caller_id,
    virtual_number,
    debug_mode=True
)

def send_init():
    __console.log('about to make config request')
    fsm_response = hermes\
        .hermes_configuration()

    __console.log('response: {}'.format(fsm_response))

    # Allows the dialplan program to access the variables
    # by processing FSM response
    hermes.set_variables_in_std(fsm_response)
    __console.log('config variables set')

def send_speech(file_descriptor):
    __console.log('send sound file to API')
    fsm_response = hermes\
        .next_action_request('AUDIO', 'FLAC', file_descriptor, conv_format='FLAC')

    __console.log('response: {}'.format(fsm_response))

    # Allows the dialplan program to access the variables
    # by processing FSM response
    hermes.set_variables_in_std(fsm_response)


def record_speech(timeout_threshold, silence_timeout_chunk, sound_array):
    merged_blocks = sound_array[:]
    _silence_ctr = 0

    while _silence_ctr <= timeout_threshold:
        raw_samples     = constants.FILE_DESCRIPTOR.read(silence_timeout_chunk)
        new_samples     = np.fromstring(raw_samples, dtype=np.int16)
        _silence_ctr    += silence_timeout_chunk

        merged_blocks = np.append(merged_blocks, new_samples)

        if not fn.is_speaking(new_samples):
            _silence_ctr = timeout_threshold + 1

    return merged_blocks


def wait_until_sound():
    _silence_ctr, samples = 0, [0]
    while fn.rms(samples) < constants.VOLUME_THRESHOLD:
        # Input Real-time Data Raw Audio from Asterisk
        raw_samples = constants.FILE_DESCRIPTOR.read(constants.CHUNK)
        samples = np.fromstring(raw_samples, dtype=np.int16)
        _silence_ctr += constants.CHUNK

        if _silence_ctr > constants.SILENCE_TIMEOUT_THRESHOLD:
            __console.log('Time Out No Speech Detected ...')
            exit_AGI()

    __console.log('Speech Detected Recording...')
    return samples


def create_flac_from(sound_samples):
    __console.log('prepare flac format for writing to file')
    n_channels, fmt     = 1, Format('flac', 'pcm16')
    caller_id           = get_stdn_var(stdin.CALLER_ID)
    __console.log('write to temp file')
    _, temp_sound_file  = mkstemp('TmpSpeechFile_' + caller_id + '.flac')
    __console.log('prepare sound file')
    flac_file           = Sndfile(temp_sound_file, 'w', fmt, n_channels, constants.RAW_RATE)

    flac_file.write_frames(np.array(sound_samples))
    __console.log('sound file saved')

    __console.log('trimming leading and trailing silence')
    y, sr = librosa.load(temp_sound_file, None)
    yt, index = librosa.effects.trim(y, top_db=constants.VOLUME_THRESHOLD)
    _, temp_sound_file_trimmed = mkstemp('TmpSpeechFile_' + caller_id + '_trimmed.flac')
    sf.write(temp_sound_file_trimmed, yt, sr, format='flac', subtype='PCM_24')
    __console.log('trimmed audio saved')

    return temp_sound_file_trimmed


def flow_handler():
    if len(session_id) == 0:
        __console.log('launch init API')
        send_init()
    else:
        __console.log('Hello Waiting For Speech')
        sound_array = wait_until_sound()

        __console.log('Recording Speech')
        sound_samples = record_speech(
            constants.SILENCE_TIMEOUT_THRESHOLD,
            constants.SILENCE_TIMEOUT_CHUNK,
            sound_array
        )

        __console.log('Writing sound to .flac file')
        sound_file = create_flac_from(sound_samples)

        __console.log('Sending .flac file to FSM')
        send_speech(sound_file)


__console.log('loaded ', __name__)
