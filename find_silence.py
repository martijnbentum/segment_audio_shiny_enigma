import audio
import numpy as np
from pathlib import Path
import soundfile as sf


def find_silences(path ='', signal = None, sr = None, frame_ms=20.0, 
    hop_ms=10.0, min_silence_seconds=1,thresh_dbfs=None, 
    adaptive_percentile=20.0, pad_frames=1, add_zero_crossings=False,
    down_sample_zero_crossings=True):
    # Main function: return long silences as list of {start, end, duration}

    if path == '':
        if signal is None or sr is None:
            raise ValueError("Either path or (signal and sr) must be provided")
        y = signal
    else:
        if not Path(path).is_file():
            raise ValueError(f"File not found: {path}")
        y, sr = audio.load_audio(path)
    frame, hop = compute_frame_hop(sr, frame_ms, hop_ms)
    frames = slice_signal(y, frame, hop)
    db = rms_dbfs(frames)
    thr = choose_threshold_dbfs(db, thresh_dbfs, adaptive_percentile)
    mask = silence_mask(db, thr)
    mask = dilate_boolean_1d(mask, pad_frames)
    time_ranges = frames_to_time_ranges(mask, frame, hop, sr, 
        min_silence_seconds)
    if add_zero_crossings:
        for time_range in time_ranges:
            start_sample = time_range['start_sample']
            end_sample = time_range['end_sample']
            zero_crossings = audio.find_all_zero_crossings(y, start_sample, 
                end_sample, strict = True, suppress_warnings = True)
            zero_crossings = zero_crossings
            if down_sample_zero_crossings and len(zero_crossings) > 10:
                step = max(1, len(zero_crossings) // 10)
                zero_crossings = zero_crossings[::step]
            time_range['zero_crossings'] = zero_crossings
    return time_ranges


def compute_frame_hop(sr, frame_ms, hop_ms):
    # Convert frame and hop sizes from ms to sample counts
    frame = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    hop = max(1, min(hop, frame))
    return frame, hop

def slice_signal(y, frame, hop):
    # Slice signal into overlapping frames (zero-pad if needed)
    if len(y) < frame:
        y = np.pad(y, (0, frame - len(y)), mode="constant")
    n_frames = 1 + int(np.ceil((len(y) - frame) / hop))
    total_needed = (n_frames - 1) * hop + frame
    if total_needed > len(y):
        y = np.pad(y, (0, total_needed - len(y)), mode="constant")
    idx = np.arange(0, n_frames * hop, hop)[:, None] + np.arange(frame)[None, :]
    return y[idx]

def rms_dbfs(frames):
    # Compute RMS energy per frame and convert to dBFS
    rms = np.sqrt(np.mean(frames**2, axis=1, dtype=np.float64)) + 1e-12
    return (20.0 * np.log10(rms)).astype(np.float32)

def choose_threshold_dbfs(dbfs, fixed_dbfs, adaptive_percentile):
    # Decide silence threshold: fixed value or adaptive percentile
    if fixed_dbfs is not None:
        return float(fixed_dbfs)
    finite = np.isfinite(dbfs)
    if not np.any(finite):
        return -np.inf
    return float(np.percentile(dbfs[finite], adaptive_percentile))

def silence_mask(dbfs, thresh_dbfs):
    # Boolean mask: True where frame energy is below threshold
    return dbfs < thresh_dbfs

def dilate_boolean_1d(mask, iterations):
    # Smooth mask by expanding silent regions by N frames
    if iterations <= 0:
        return mask
    k = 2 * iterations + 1
    conv = np.convolve(mask.astype(np.int32), np.ones(k, dtype=np.int32), mode="same")
    return conv > 0

def segment_bounds(mask):
    # Find contiguous silent regions as (start_frame, end_frame)
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
    # Keep only silences above minimum duration, convert to seconds
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
    # Wrapper: convert mask → frame bounds → filtered time ranges
    bounds = segment_bounds(mask)
    return filter_and_convert_segments(bounds, frame, hop, sr, min_silence_sec)

