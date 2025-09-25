import copy
import json
import librosa
from pathlib import Path
from progressbar import progressbar
import soundfile as sf

def handle_audio(filename, segment_duration = 10, n_segments = 10,
    sample_rate = 16000):
    y, sr = load_audio(filename, sr = sample_rate)
    segments = segment_audio(y, sample_rate, segment_duration, 
        n_segments = n_segments)

def load_audio(filename, sr=None, mono=True):
    y, sr = librosa.load(filename, sr=sr, mono=mono)
    return y, sr


def segment_audio(filename, sample_rate = 16000, segment_duration = 30, 
     n_segments=10, drop_last=False, save = True):

    segment_length = int(sample_rate * segment_duration)
    y, _ = load_audio(filename, sr=sample_rate)
    total_samples = len(y)

    segments = []
    start_sample = 0
    index = 1
    n = int((total_samples / segment_length)) + 1
    for _ in progressbar(range(n)):
        if n_segments is not None and index > n_segments:
            print(f"Reached max number of segments: {n_segments}")
            break
        start_sample, end_sample = get_start_and_end_sample_indices(
            start_sample, segment_length, y)
        if end_sample >= total_samples and drop_last:
            print("Rreached end of the file. Dropping last segment")
            print("Total segments created:", len(segments))
            break
        segment_filename = make_segment_filename(filename, index, start_sample,
            end_sample)
        segment = make_segment(y, filename, segment_filename, index,
            start_sample, end_sample, sample_rate)
        segments.append(segment)
        start_sample = end_sample
        index += 1
    
    if save:
        save_segments(segments, sample_rate, filename)
    return segments

def save_segments(segments, sr, filename, output_dir=None, subtype="PCM_16"):
    p = Path(filename)
    if output_dir is None:
        output_dir = Path(p.stem + "_segments")
        output_dir.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        output_filename = output_dir / segment["segment_filename"]
        audio = segment["audio_segment"]
        sf.write(output_filename, audio, sr, subtype=subtype)
        print(f"Saved segment: {output_filename}")
    print(f"Saved {len(segments)} segments to {output_dir}")
    save_segments_json(segments, output_dir)

def save_segments_json(segments, output_dir):
    segments = copy.deepcopy(segments)
    for segment in segments:
        del segment['audio_segment'] 
    audio_filename = segments[0]["filename"]
    n_segments = len(segments)
    duration = int(round(sum([s["duration"] for s in segments]) / n_segments))
    f = Path(output_dir) / f"{Path(audio_filename).stem}"
    json_filename = str(f) + f"_segments-{n_segments}_duration-{duration}.json"
    with open(json_filename, "w") as f:
        json.dump(segments, f, indent=4)
    print(f"Saved segments metadata to {json_filename}")

def get_start_and_end_sample_indices(start_sample, segment_length, y):
    total_samples = len(y)
    target_end = start_sample + segment_length
    if target_end >= total_samples:
        return start_sample, target_end 
    end_sample = next_zero_crossing(y, target_end)
    if end_sample <= start_sample:  # safety net
        m = f"end_sample ({end_sample}) <= start_sample ({start_sample})"
        raise ValueError(m)
    return start_sample, end_sample 

def make_segment(y, filename, segment_filename, segment_index, start_sample, 
    end_sample, sample_rate):
    segment = {
        "filename": filename,
        "segment_filename": segment_filename,
        "segment_index": segment_index,
        "start": start_sample / sample_rate,
        "end": end_sample / sample_rate,
        "duration": (end_sample - start_sample) / sample_rate,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "sample_rate": sample_rate,
        "audio_segment": y[start_sample:end_sample],
    }
    return segment

def make_segment_filename(filename, segment_index, start_sample, end_sample):
    p = Path(filename)
    segment_filename = f"{p.stem}_n-{segment_index}"
    # segment_filename += f"_s-{start_sample}_e-{end_sample}.wav"
    segment_filename += ".wav"
    return segment_filename


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
