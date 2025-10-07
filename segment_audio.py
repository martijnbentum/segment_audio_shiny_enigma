import audio
import copy
import json
from pathlib import Path
from progressbar import progressbar
import utils 


def segment_audio(audio_filename, tag = None, segment_duration = 30, 
    start_time = None, end_time = None, n_segments = None, 
    sample_rate = 16000, drop_last = True, save = False, overwrite = False):
    if not overwrite and save:
        output_dir, exists = utils.make_output_directory_name(audio_filename)
        if exists:
            m = f'Output directory {output_dir} exists, '
            m += 'use overwrite=True to overwrite'
            raise ValueError(m)

    segment_length = int(sample_rate * segment_duration)
    y, _ = audio.load_audio(audio_filename, sr=sample_rate)
    total_samples = len(y)

    start_time, end_time, start_sample, final_sample = utils.handle_start_end(
        start_time=start_time, end_time=end_time, audio_filename=audio_filename,
        total_n_samples=total_samples, y=y, sample_rate=sample_rate,
        tag = tag, overwrite=overwrite)
    print(f'start_time: {start_time}, end_time: {end_time}, '
        f'start_sample: {start_sample}, final_sample: {final_sample}, '
        f'total_samples: {total_samples}')
        

    segments = []
    index = 1
    n = int((total_samples / segment_length)) + 1
    for _ in progressbar(range(n)):
        if n_segments is not None and index > n_segments:
            print(f"Reached max number of segments: {n_segments}")
            break
        start_sample, end_sample = get_start_and_end_sample_indices(
            start_sample, segment_length, y)
        if end_sample >= final_sample and drop_last:
            print("Reached end of the file. Dropping last segment")
            break
        segment_filename = make_segment_filename(audio_filename, tag, index, 
            start_sample, end_sample)
        segment = make_segment(y, audio_filename, segment_filename, index,
            start_sample, end_sample, sample_rate)
        segments.append(segment)
        start_sample = end_sample
        index += 1
    if n_segments is not None:
        print(f"Created {len(segments)} segments, max was {n_segments}")
    else:
        print(f"Created {len(segments)} segments")
    if save:
        save_segments(segments, sample_rate, audio_filename, tag, overwrite)
    return segments

def save_segments(segments, sr, audio_filename, tag, overwrite = False, 
    subtype="PCM_16"):
    p = Path(audio_filename)
    output_dir = utils.handle_output_directory(audio_filename, tag,
        overwrite=overwrite)
    for segment in segments:
        output_filename = output_dir / segment["segment_filename"]
        y = segment["audio_segment"]
        #sf.write(output_filename, audio, sr, subtype=subtype)
        audio.save_audio(y, output_filename, sr, subtype = subtype)
        print(f"Saved segment: {output_filename}")
    print(f"Saved {len(segments)} segments to {output_dir}")
    save_segments_json(segments, output_dir)

def save_segments_json(segments, output_dir):
    segments = copy.deepcopy(segments)
    for segment in segments:
        del segment['audio_segment'] 
    audio_filename = segments[0]["audio_filename"]
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
    end_sample = audio.next_zero_crossing(y, target_end)
    if end_sample <= start_sample:  # safety net
        m = f"end_sample ({end_sample}) <= start_sample ({start_sample})"
        raise ValueError(m)
    return start_sample, end_sample 

def make_segment(y, audio_filename, segment_filename, segment_index, 
    start_sample, end_sample, sample_rate):
    if end_sample > len(y):
        end_sample = len(y)
        shorter = True
    else: shorter = False
    if start_sample >= len(y):
        raise ValueError(f"start_sample >= len(y), {start_sample} >= {len(y)}")
    segment = {
        "audio_filename": audio_filename,
        "segment_filename": segment_filename,
        "segment_index": segment_index,
        "start": start_sample / sample_rate,
        "end": end_sample / sample_rate,
        "duration": (end_sample - start_sample) / sample_rate,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "sample_rate": sample_rate,
        "audio_segment": y[start_sample:end_sample],
        "shorter": shorter
    }
    return segment

def make_segment_filename(audio_filename, tag, segment_index, start_sample, 
    end_sample):
    p = Path(audio_filename)
    segment_filename = f"{p.stem}"
    if tag is not None:
        segment_filename += f"_{tag}"
    segment_filename += f"_n-{segment_index}"
    segment_filename += ".wav"
    return segment_filename


