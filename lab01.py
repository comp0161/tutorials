#!/usr/bin/env/python
# -*- coding: utf-8 -*-

"""
Basic audio generation for COMP0161 week 1 tutorial.

This is primarily intended for demoing waveforms during
the session and may not have much applicability beyond
that. It's also not exactly a model of high class coding.
"""

import sys, os, os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import simpleaudio as sa

rng = np.random.default_rng()

DURATION = 2
SAMPLE_RATE = 44100
PITCH = 110
PITCHES = [55, 110, 220, 440, 880]

def play(x, rate=SAMPLE_RATE, dupe_stereo=True, scale=1):
    """
    Play a numpy array using simpleaudio.
    Data is scaled to range and converted to the
    required int16 type, and optionally (on by
    default) duplicated to stereo if mono.
    """
    x = ((x * 32767 * scale) / np.max(np.abs(x))).astype(np.int16)
    if dupe_stereo and (len(x.shape) == 1):
        x = np.stack((x,x)).T.copy()
    
    return sa.play_buffer(x, 2, 2, rate)


def tone(hz=PITCH, duration=DURATION, rate=SAMPLE_RATE, phase=0):
    """
    Generate a simple pure sine tone of specified frequency
    and duration.
    """
    return np.sin(phase + hz * 2 * np.pi * np.arange(duration * rate)/rate)

def tones(hz=PITCHES, duration=DURATION, weights=None, rate=SAMPLE_RATE, phases=None):
    """
    Generate a (potentially weighted) sum of pure sine tones.
    """
    if weights is None:
        weights = np.ones(len(hz))
    
    if phases is None:
        phases = np.zeros(len(hz))
    
    result = weights[0] * tone(hz[0], duration, rate, phases[0])
    for ii in range(1, len(hz)):
        result += weights[ii] * tone(hz[ii], duration, rate, phases[ii])
    
    return result

def saw (hz=PITCH, duration=DURATION, rate=SAMPLE_RATE, max_harm=10000, random_phase=0 ):
    """
    Generate a band-limited sawtooth wave of specified frequency
    and duration. Phases of the individual harmonics can be randomised
    by an amount specified by`random_phase` (interpreted as
    largest fraction of a cycle).
    """
    assert(hz < rate/2)
    
    angles = 2 * np.pi * np.arange(duration * rate)/rate
    if max_harm is None: max_harm = 10000
    max_harm = np.min([int(np.floor(rate / (2 * hz))), max_harm])
    
    # start with the fundamental
    result = -np.sin(angles * hz)
    components = 1
    
    # add the harmonics
    for harm in range(2, max_harm + 1):
        result -= np.sin(angles * hz * harm + rng.uniform() * random_phase * 2 * np.pi ) / harm
        components += 1
    
    # scale into [-1, 1]
    result = 2 * (result - np.min(result))/(np.max(result) - np.min(result)) - 1

    print(f'\n  saw ==> max_harm: {max_harm}, components: {components}, max_freq: {hz * harm}')
    
    return result

def square (hz=PITCH, duration=DURATION, rate=SAMPLE_RATE, max_harm=10000, random_phase=0 ):
    """
    Generate a band-limited square wave of specified frequency
    and duration. Phases of the individual harmonics can be randomised
    by an amount specified by`random_phase` (interpreted as
    largest fraction of a cycle).
    """
    assert(hz < rate/2)
    
    angles = 2 * np.pi * np.arange(duration * rate)/rate
    
    if max_harm is None: max_harm = 10000
    max_harm = np.min([int(np.floor(rate / (2 * hz))), max_harm])
    
    # start with the fundamental
    result = np.sin(angles * hz)
    
    components = 1
    
    # add the harmonics
    for harm in range(3, max_harm + 1, 2):
        result += np.sin(angles * hz * harm + rng.uniform() * random_phase * 2 * np.pi ) / harm
        components += 1
    
    # scale into [-1, 1]
    result = 2 * (result - np.min(result))/(np.max(result) - np.min(result)) - 1
    
    print(f'\n  square ==> max_harm: {max_harm}, components: {components}, max_freq: {hz * harm}')
    
    return result

def tri (hz=PITCH, duration=DURATION, rate=SAMPLE_RATE, max_harm=10000, random_phase=0 ):
    """
    Generate a band-limited triangle wave of specified frequency
    and duration. Phases of the individual harmonics can be randomised
    by an amount specified by`random_phase` (interpreted as
    largest fraction of a cycle).
    """
    assert(hz < rate/2)
    
    angles = 2 * np.pi * np.arange(duration * rate)/rate
    
    if max_harm is None: max_harm = 10000
    max_harm = np.min([int(np.floor(rate / (2 * hz))), max_harm])
    
    # start with the fundamental
    result = np.sin(angles * hz)
    components = 1
    
    # add the harmonics
    direction = -1
    for harm in range(3, max_harm + 1, 2):
        result += direction * np.sin(angles * hz * harm + rng.uniform() * random_phase * 2 * np.pi ) / (harm * harm)
        direction = -direction
        components += 1
    
    # scale into [-1, 1]
    result = 2 * (result - np.min(result))/(np.max(result) - np.min(result)) - 1

    print(f'\n  tri ==> max_harm: {max_harm}, components: {components}, max_freq: {hz * harm}')
    
    return result

DEMOS = [
    ("Sine wave at 110 Hz", lambda: play(tone())),
    ("Sine wave at 220 Hz", lambda: play(tone(220))),
    ("Sine wave at 440 Hz", lambda: play(tone(440))),
    ("Sum of equal sines at 110, 220", lambda: play(tones([110,220]))),
    ("Sum of unequal sines 110, 220", lambda: play(tones([110,220], weights=[0.3,0.7], phases=[0, np.pi]))),
    ("Sum of unequal sines 110, 220, 330", lambda: play(tones([110,220,330], weights=[0.2,0.3,0.5], phases=[0, np.pi/2, 2*np.pi/3]))),
    
    ("Saw with 2 harmonics", lambda: play(saw(max_harm=2))),
    ("Saw with 3 harmonics", lambda: play(saw(max_harm=3))),
    ("Saw with 5 harmonics", lambda: play(saw(max_harm=5))),
    ("Saw with 10 harmonics", lambda: play(saw(max_harm=10))),
    ("Saw with all harmonics", lambda: play(saw())),
    
    ("Square with 2 harmonics", lambda: play(square(max_harm=3))),
    ("Square with 3 harmonics", lambda: play(square(max_harm=5))),    
    ("Square with 5 harmonics", lambda: play(square(max_harm=9))),
    ("Square with 10 harmonics", lambda: play(square(max_harm=19))),
    ("Square with all harmonics", lambda: play(square())),
    
    ("Triangle with 2 harmonics", lambda: play(tri(max_harm=3))),
    ("Triangle with 3 harmonics", lambda: play(tri(max_harm=5))),    
    ("Triangle with 5 harmonics", lambda: play(tri(max_harm=9))),
    ("Triangle with 10 harmonics", lambda: play(tri(max_harm=19))),
    ("Triangle with all harmonics", lambda: play(tri())),
    
    ("Saw with 5 harmonics, random phases", lambda: play(saw(max_harm=5, random_phase=1))),
    ("Saw with 10 harmonics, random phases", lambda: play(saw(max_harm=10, random_phase=1))),
    ("Saw with all harmonics, random phases", lambda: play(saw(random_phase=1))),

    ("Square with 5 harmonics", lambda: play(square(max_harm=9, random_phase=1))),
    ("Square with 10 harmonics", lambda: play(square(max_harm=19, random_phase=1))),
    ("Square with all harmonics", lambda: play(square(random_phase=1))),

    ("Triangle with 5 harmonics", lambda: play(tri(max_harm=9, random_phase=1))),
    ("Triangle with 10 harmonics", lambda: play(tri(max_harm=19, random_phase=1))),
    ("Triangle with all harmonics", lambda: play(tri(random_phase=1))),
]

def demo():
    """
    Simple looping shortcut menu for triggering some
    examples in tutorial 1. 
    """
    while True:
        print()
        for ii, (label, _) in enumerate(DEMOS):
            print(f'{ii}: {label}')
        
        opt = input("Choose option: ")
        
        if opt:
            try:
                DEMOS[int(opt)][1]()
            except:
                print('not recognised')
        else:
            break
