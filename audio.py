import json
import librosa
from pathlib import Path
import soundfile as sf

def handle_audio(filename, segment_duration = 10, n_segments = 10,
    sample_rate = 16000):
    y, sr = load_audio(filename, sr = sample_rate)
    segments = segment_audio(y, sample_rate, segment_duration, 
        n_segments = n_segments)

def load_audio(filename, sr=None, mono=True):
    y, sr = librosa.load(filename, sr=sr, mono=mono)
    return y, sr



def segment_audio(filename, sample_rate, segment_duration, 
     n_segments=None, drop_last=False):
    segment_length = int(sample_rate * segment_duration)
    y, _ = load_audio(filename, sr=sample_rate)
    total_samples = len(y)

    segments = []
    start_sample = 0
    index = 1

    while start_sample < total_samples:
        if n_segments is not None and index > n_segments:
            break

        target_end = start_sample + segment_length
        if target_end >= total_samples:
            if drop_last and target_end > total_samples:
                break
            end_sample = total_samples
        else:
            end_sample = next_zero_crossing(y, target_end)

        if end_sample <= start_sample:  # safety net
            end_sample = min(start_sample + segment_length, total_samples)
        segment_filename = make_segment_filename(filename, index, start_sample, 
            end_sample)

        segment = {
            "filename": filename,
            "segment_filename": segment_filename,
            "segment_index": index,
            "start": start_sample / sample_rate,
            "end": end_sample / sample_rate,
            "duration": (end_sample - start_sample) / sample_rate,
            "start_sample": start_sample,
            "end_sample": end_sample,
            "sample_rate": sample,
            "audio_segment": y[start_sample:end_sample],
        }

        start_sample = end_sample
        index += 1
    return segments

def make_segment_filename(filename, segment_index, start_sample, end_sample):
    p = Path(filename)
    segment_filename = f"{p.stem}_n-{segment_index}"
    segment_filename += f"_start-{start_sample}_end-{end_sample}.wav"
    return filename
    

def save_segments(segments, sr, filename, output_dir=None, subtype="PCM_16"):
    p = Path(filename)
    out_dir = Path(output_dir) if output_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    output_filename = 
    saved = []
    for seg in segments:
        out_path = out_dir / name
        sf.write(out_path, seg["y"], sr, subtype=subtype)
        saved.append(out_path)
    return saved


def next_zero_crossing(y, start_idx):
    n = len(y)
    if start_idx <= 0:
        return 0
    i = min(start_idx, n - 1)
    while i < n:
        if y[i] == 0.0:
            return i
        prev, cur = y[i - 1], y[i]
        if (prev <= 0 and cur > 0) or (prev >= 0 and cur < 0):
            return i
        i += 1
    return n
