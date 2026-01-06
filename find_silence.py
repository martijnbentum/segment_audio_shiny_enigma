import audio
import librosa
import math
import numpy as np
from pathlib import Path
import soundfile as sf

def find_closest_silence_with_sample_index(sample_index, silences, sr, 
    max_distance = 60, ensure_zero_crossing = True):
    '''Find the closest silence to the given sample index.
    sample_index:          timepoint in sample index (integer)
    silences:              list of silences as returned by find_silences()
    sr:                    sample rate (integer)
    max_distance:          maximum distance (in seconds) to consider a silence
    ensure_zero_crossing:  if True, only consider silences that have zero crossings
    '''
        
    return find_closest_silence(sample_index / sr, silences, sr, max_distance,
        ensure_zero_crossing)

def find_closest_silence(seconds, silences, sr, max_distance = 60, 
    ensure_zero_crossing = True):
    '''Find the closest silence to the given timepoint in seconds.
    seconds:               timepoint in seconds (float)
    silences:              list of silences as returned by find_silences()
    sr:                    sample rate (integer)
    max_distance:          maximum distance (in seconds) to consider a silence
    ensure_zero_crossing:  if True, only consider silences that have zero crossings
    '''
    start_distances = [abs(seconds - s['start']) for s in silences]
    end_distances = [abs(seconds - s['end']) for s in silences]
    shortest_distance = math.inf
    for i, (s, e) in enumerate(zip(start_distances, end_distances)):
        silence = silences[i]
        start_time = silence['start']
        end_time = silence['end']
        if ensure_zero_crossing:
            has_zero_crossings = silence.get('has_zero_crossings', False)
            if not has_zero_crossings:
                m = f"Skipping silence at {silence['start']} - "
                m += f"{silence['end']} seconds: "
                m += "no zero crossings found"
                print(m)
                continue
        if seconds > start_time and seconds < end_time:
            # Timepoint is inside this silence
            shortest_distance = 0
            closest_index = i
            break
        d = min(s, e)
        if d < shortest_distance:
            shortest_distance = d
            closest_index = i
    if max_distance <= shortest_distance:
        m = f"No silence within {max_distance} seconds"
        m += f", closest is {shortest_distance} seconds away"
        m += f" for time {seconds}"
        raise ValueError(m)
    silence = silences[closest_index]
    zero_crossings = silence['zero_crossings']
    closest_sample = find_closest_sample(zero_crossings, seconds= seconds, 
        sr=sr, collar_seconds = 0.6)
    silence['closest_zero_crossing'] = closest_sample
    silence['zero_crossings'] = downsample_sequence(zero_crossings, 10, 
        closest_sample)
    return silence


def find_silences(path ='', signal = None, sr = None, frame_ms=20.0, 
    hop_ms=10.0, min_silence_seconds=1,thresh_dbfs=None, 
    adaptive_percentile=20.0, pad_frames=1, add_zero_crossings=False,
    down_sample_zero_crossings=False, remove_long_silences=True):
    '''Find silences in an audio recording.
    Either path to audio file or signal and sr must be provided.
    path:                        path to audio file (string)
    signal:                      audio signal as numpy array (1D)
    sr:                          sample rate of the signal (integer)
    frame_ms:                    frame size in milliseconds (float)
    hop_ms:                      hop size in milliseconds (float)
    min_silence_seconds:         minimum duration of silence to report (float)
    thresh_dbfs:                 fixed silence threshold in dBFS (float or None)
    adaptive_percentile:         if thresh_dbfs is None, use this percentile of
                                 all frame energies as the silence threshold 
                                 (float)
    pad_frames:                  number of frames to pad silent regions 
                                 (integer)
    add_zero_crossings:          if True, find zero crossings in each silence 
                                 (bool)
    down_sample_zero_crossings:  if True, down-sample zero crossings to max 10 
                                 (bool)
    remove_long_silences:        if True, remove long silences before finding 
                                 threshold (bool)
    '''
    # Main function: return long silences as list of {start, end, duration}

    if path == '':
        if signal is None or sr is None:
            raise ValueError("Either path or (signal and sr) must be provided")
        y = signal
    else:
        if not Path(path).is_file():
            raise ValueError(f"File not found: {path}")
        y, sr = audio.load_audio(path)
    thr = find_threshold(y, sr, frame_ms, hop_ms, thresh_dbfs,
        adaptive_percentile, remove_long_silences)

    frame, hop = compute_frame_hop(sr, frame_ms, hop_ms)
    frames = slice_signal(y, frame, hop)
    db = rms_dbfs(frames)
    print(f"number of frames: {len(frames)}")
    print(f"duration (seconds): {len(y) / sr}")
    print(f"min dbFS: {np.min(db)}, max dbFS: {np.max(db)}")
    print(f"Using silence threshold: {thr} dBFS")

    mask = silence_mask(db, thr)
    mask = dilate_boolean_1d(mask, pad_frames)
    time_ranges = frames_to_time_ranges(mask, frame, hop, sr, 
        min_silence_seconds)
    print(f"Found {len(time_ranges)} silences longer than {min_silence_seconds}")
    if add_zero_crossings:
        for time_range in time_ranges:
            start_sample = time_range['start_sample']
            end_sample = time_range['end_sample']
            zero_crossings = audio.find_all_zero_crossings(y, start_sample, 
                end_sample, strict = True, suppress_warnings = True)
            if down_sample_zero_crossings and len(zero_crossings) > 10:
                step = max(1, len(zero_crossings) // 10)
                zero_crossings = zero_crossings[::step]
            time_range['zero_crossings'] = zero_crossings
            time_range['has_zero_crossings'] = len(zero_crossings) > 0
    else: 
        for time_range in time_ranges:
            time_range['has_zero_crossings'] = None
            time_range['zero_crossings'] = None
    return time_ranges


def compute_frame_hop(sr, frame_ms, hop_ms):
    '''Compute frame and hop sizes in samples from milliseconds
    '''
    frame = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    hop = max(1, min(hop, frame))
    return frame, hop

def slice_signal(y, frame, hop):
    '''Slice signal into overlapping frames (if hop < frame).
    '''
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)), mode="constant")
    n_frames = 1 + int(np.ceil((len(y) - frame) / hop))
    total_needed = (n_frames - 1) * hop + frame
    if total_needed > len(y):
        y = np.pad(y, (0, total_needed - len(y)), mode="constant")
    idx = np.arange(0, n_frames * hop, hop)[:, None] + np.arange(frame)[None, :]
    return y[idx]

def rms_dbfs(frames):
    '''Compute RMS in dBFS for each frame.
    frames:   2D numpy array: (n_frames, frame_size)
    '''
    rms = np.sqrt(np.mean(frames**2, axis=1, dtype=np.float64)) + 1e-12
    return (20.0 * np.log10(rms)).astype(np.float32)

def choose_threshold_dbfs(dbfs, fixed_dbfs, adaptive_percentile):
    '''Decide silence threshold: fixed value or adaptive percentile.
    dbfs:                 1D numpy array of frame energies in dBFS
    fixed_dbfs:           fixed threshold in dBFS (float or None)
    adaptive_percentile:  percentile to use if fixed_dbfs is None (float)
                          should be between 0 and 100 
    '''
    if fixed_dbfs is not None:
        return float(fixed_dbfs)
    finite = np.isfinite(dbfs)
    if not np.any(finite):
        return -np.inf
    return float(np.percentile(dbfs[finite], adaptive_percentile))

def silence_mask(dbfs, thresh_dbfs):
    ''' Boolean mask: True where frame energy is below threshold.
    '''
    return dbfs < thresh_dbfs

def dilate_boolean_1d(mask, iterations):
    ''' Smooth mask by expanding silent regions by N frames.
    iterations:  number of frames to expand on each side (integer)
    '''
    if iterations <= 0:
        return mask
    k = 2 * iterations + 1
    conv = np.convolve(mask.astype(np.int32), np.ones(k, dtype=np.int32), 
        mode="same")
    return conv > 0

def segment_bounds(mask):
    '''Find contiguous silent regions as (start_frame, end_frame).
    '''
    starts, ends = [], []
    inside = False
    for i, val in enumerate(mask):
        if val and not inside:
            starts.append(i)
            inside = True
        elif not val and inside:
            ends.append(i - 1)
            inside = False
    if inside:
        ends.append(len(mask) - 1)
    return list(zip(starts, ends))

def filter_and_convert_segments(bounds, frame, hop, sr, min_silence_sec):
    '''Keep only silences above minimum duration, convert to seconds.
    bounds:            list of (start_frame, end_frame) tuples
    frame:             frame size in samples (integer)
    hop:               hop size in samples (integer)
    sr:                sample rate (integer)
    min_silence_sec:  minimum silence duration in seconds (float)
    '''
    min_frames = int(np.ceil(min_silence_sec * sr / hop))
    segments = []
    for start_f, end_f in bounds:
        length = end_f - start_f + 1
        if length >= min_frames:
            start_t = (start_f * hop) / sr
            end_t = (end_f * hop + frame) / sr
            segments.append({
                "start": round(start_t, 3),
                "end": round(end_t, 3),
                "duration": round(end_t - start_t, 3),
                "start_sample": start_f * hop, 
                "end_sample": end_f * hop + frame,
            })
    return segments

def frames_to_time_ranges(mask, frame, hop, sr, min_silence_sec):
    ''' Wrapper: convert mask → frame bounds → filtered time ranges. 
    mask:              1D boolean numpy array
    frame:             frame size in samples (integer)
    hop:               hop size in samples (integer)
    sr:                sample rate (integer)
    min_silence_sec:  minimum silence duration in seconds (float)
    '''
    bounds = segment_bounds(mask)
    return filter_and_convert_segments(bounds, frame, hop, sr, min_silence_sec)


def remove_long_silences_with_librosa(y, sr, top_db=60, frame_length=2048, 
    hop_length=512):
    '''Remove long silences from the audio signal using librosa.effects.split.
    y:         1D numpy array: audio signal
    sr:        sample rate (integer)
    top_db:    threshold (in dB) below reference to consider as silence
    frame_length:  frame length for analysis (integer)
    hop_length:    hop length for analysis (integer)
    '''
    intervals = librosa.effects.split(y, top_db=top_db, 
        frame_length=frame_length, hop_length=hop_length)
    new_signal = concatenate_intervals(y, intervals)
    print("Removing long silences with librosa.effects.split")
    print(f"original signal duration (seconds): {len(y) / sr}") 
    print(f"found {len(intervals)} non-silent intervals")
    print(f"removed duration (seconds): {(len(y) - len(new_signal)) / sr}")
    print(f"new signal duration (seconds): {len(new_signal) / sr}")
    return new_signal
    

def concatenate_intervals(signal, intervals):
    '''Concatenate segments of the signal defined by intervals.
    signal:     1D numpy array: audio signal
    intervals:  list of (start_sample, end_sample) tuples
    '''
    return np.concatenate([signal[start:end] for start, end in intervals])

def find_threshold(signal, sr, frame_ms=20.0, hop_ms=10.0, thresh_dbfs = None,
    adaptive_percentile = 20.0, remove_long_silences=True):
    '''Find silence threshold in dBFS for the given signal.
    signal:                      audio signal as numpy array (1D)
    sr:                          sample rate of the signal (integer)
    frame_ms:                    frame size in milliseconds (float)
    hop_ms:                      hop size in milliseconds (float)
    thresh_dbfs:                 fixed silence threshold in dBFS (float or None)
    adaptive_percentile:         if thresh_dbfs is None, use this percentile 
                                 should be between 0 and 100 (float)
    '''
    if remove_long_silences:
        print("Removing long silences ", 
            "set remove_long_silences_with_librosa=False to disable")
        signal = remove_long_silences_with_librosa(signal, sr)
    frame, hop = compute_frame_hop(sr, frame_ms, hop_ms)
    frames = slice_signal(signal, frame, hop)
    db = rms_dbfs(frames)
    thr = choose_threshold_dbfs(db, thresh_dbfs, adaptive_percentile)
    print(f"number of frames: {len(frames)}")
    print(f"min dbFS: {np.min(db)}, max dbFS: {np.max(db)}")
    print(f"Using silence threshold: {thr} dBFS")
    return thr

def downsample_sequence(sequence, target_length, obligatory_item = None):
    '''Down-sample a sequence to the target length, 
    ensuring obligatory_item is included.
    sequence:          list of items
    target_length:     desired length of the output list (integer)
    obligatory_item:   item that must be included in the output list
    '''
    if len(sequence) <= target_length:
        return sequence
    step = len(sequence) / target_length
    indices = [int(i * step) for i in range(target_length)]
    output = [sequence[i] for i in indices]
    if obligatory_item is not None and obligatory_item not in output:
        # Ensure obligatory item is included
        for i in range(len(output)):
            if (i == 0 or output[i-1] < obligatory_item) and \
               (i == len(output)-1 or output[i+1] > obligatory_item):
                output[i] = obligatory_item
                break
    return output
        

def find_closest_sample(samples, seconds = None, sample = None, sr = None,
    collar_seconds = None):
    '''Find the closest sample in samples to the given timepoint 
    (in seconds or sample index).
    samples:        list of sample indices (integers)
    seconds:        timepoint in seconds (float)
    sample:         timepoint in sample index (integer)
    sr:             sample rate (required if seconds /collar_seconds is provided)
    collar_seconds: if provided, adjust the sample to be within
                    collar_seconds of the edges of the samples range.
    '''
    if seconds is None and sample is None:
        raise ValueError("Either seconds or sample must be provided")
    if seconds is not None:
        if sr is None:
            raise ValueError("sr must be provided when sample is provided")
        sample = int(seconds * sr)
    orginal_sample = sample
    if collar_seconds is not None:
        if sr is None:
            raise ValueError("sr must be provided when collar_seconds is provided")
        collar_sample = int(collar_seconds * sr)
        first, last = samples[0], samples[-1]
        adjusted = False
        temp_closest_sample = min(samples, key=lambda x: abs(x - sample))
        if (last - first) <= (2 * collar_sample):
            adjusted = True
            sample = (first + last) // 2
        elif sample < (first + collar_sample) and sample < (last - collar_sample):
            adjusted = True
            sample = first + collar_sample
        elif sample > (last - collar_sample) and sample > (first + collar_sample):
            adjusted = True
            sample = last - collar_sample
        if adjusted:
            adjusted_seconds = (sample - temp_closest_sample) / sr
            print(f"Adjusted with collar: {sample}, original: {orginal_sample}")
            print(f"Adjusted with collar: {adjusted_seconds} seconds ")
    closest_sample = min(samples, key=lambda x: abs(x - sample))
    if orginal_sample != closest_sample:
        adjusted_seconds = (closest_sample - orginal_sample) / sr
        print(f"adjusted sample from {orginal_sample} to {closest_sample}")
        print(f"adjusted by {adjusted_seconds} seconds")
    return closest_sample

