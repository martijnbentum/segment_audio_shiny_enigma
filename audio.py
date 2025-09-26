import copy
import json
import librosa
from pathlib import Path
from progressbar import progressbar
import math
import soundfile as sf


def load_audio(filename, sr=None, mono=True):
    print(f"Loading audio file: {filename}")
    y, sr = librosa.load(filename, sr=sr, mono=mono)
    return y, sr

def save_audio(y, filename, sr, subtype="PCM_16"):
    print(f"Saving audio to {filename}")
    sf.write(filename, y, sr, subtype=subtype)

def next_zero_crossing(y, start_index = 0, end_index = None, 
    coarse_match = False, strict = False):
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if end_index is None:
        end_index = len(y) 
    i = min(start_index, end_index)
    while i < end_index:
        if math.isclose(y[i], 0.0, abs_tol= 1e-6):
            return i
        prev, cur = y[i - 1], y[i]
        if coarse_match:
            if (prev <= 0 and cur > 0) or (prev >= 0 and cur < 0):
                return i
        i += 1
    if start_index / len(y) > 0.99:
        m=f"{start_index} start index near the end of the signal" 
        m += " did not find zero crossing"
        raise ValueError(m)
    if not coarse_match and not strict:
        print(f"{start_index} No exact zero crossing found, trying coarse match")
        return next_zero_crossing(y, start_index, coarse_match = True)
    raise ValueError("No zero crossing found")

def find_all_zero_crossings(y, start_index = 0, end_index = None,
    strict = False, verbose = True, suppress_warnings = False):
    zero_crossings = []
    if end_index is None:
        end_index = len(y)
    last_index = start_index
    while last_index < end_index: 
        try:
            zc = next_zero_crossing(y, last_index, end_index, strict = strict) 
            zero_crossings.append(zc)
            last_index = zc + 1
        except ValueError:
            break
    distance_to_end = end_index - last_index
    perc_signal_after_last = 100 * distance_to_end / end_index
    if (verbose or perc_signal_after_last > 5.0) and not suppress_warnings:
        m = f"Found {len(zero_crossings)} zero crossings \n"
        m += f"Distance from last zero crossing to end: "
        m += f"{distance_to_end} samples\n"
        m += f"Total samples: {end_index}\n" 
        m += f"percentage of signal after last zero crossing: "
        m += f"{perc_signal_after_last:.2f}%"
        print(m)
    return zero_crossings
        
