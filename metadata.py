import audio
import copy
import json
import utils
from pathlib import Path

metadata_dir = Path('metadata/')

def audio_filename_to_summary(audio_filename):
    segment_directories = audio_filename_to_ordered_segment_directories(
        audio_filename)
    end_sample, _ = audio_filename_to_end_sample(audio_filename)
    metadatas = []
    total_duration = 0.0
    tags = []
    segment_durations = []
    n_segments = []
    print('\n' + '-' * 40)
    print(f"Summary for: {audio_filename}")
    print(f"Found {len(segment_directories)} segment directories") 
    print(f"directories: \n"+ '\n'.join(list(map(str,segment_directories))))
    print('\n' + '-' * 40)
    for d in segment_directories:
        metadata = segment_dir_to_metadata(d)
        tags.append(metadata['tag'])
        segment_durations.append(metadata['segment_duration'])
        n_segments.append(metadata['n_segments'])
        total_duration += metadata['duration']
        summary = metadata_to_summary(metadata)
        print(f"Segment directory: {d}")
        print(summary)
        print("-" * 40)
    m = f"Audio filename: {audio_filename}\n"
    m += f"Tags: {tags}\n"
    m += f"Number of segments: {n_segments}\n"
    m += f"Segment durations: {segment_durations}\n"
    m += f"Total duration across {len(segment_directories)} "
    m += f"segment directories: {total_duration:2f} seconds\n"
    m += f"End sample of last segment: {end_sample}\n"
    end_time = end_sample / metadata['sample_rate']
    m += f"End time of last segment: {end_time:2f} seconds\n"
    print(m)
    
def metadata_to_summary(metadata):
    m = f"filename: {metadata['audio_filename']}\n"
    m += f"tag: {metadata['tag']}\n"
    m += f'start sample: {metadata["start_sample"]}, '
    m += f'end sample: {metadata["end_sample"]}\n'
    m += f"start time: {metadata['start_time']},  "
    m += f"end time: {metadata['end_time']}, "
    m += f"duration: {metadata['duration']} seconds\n"
    m += f"n_segments: {metadata['n_segments']}\n"
    m += f"segment duration: {metadata['segment_duration']} seconds\n"
    return m
    
def segment_dir_to_metadata(segment_dir = None, json_filename = None):
    '''load segment metadata from a json file for a given segment directory
    '''
    if json_filename is not None:
        if Path(json_filename).is_file() is False:
            if segment_dir is None:
                raise ValueError(f"{json_filename} is not a file")
            json_filename = Path(segment_dir) / json_filename
        if not Path(json_filename).is_file():
            raise ValueError(f"{json_filename} is not a file")
    elif segment_dir is not None:
        p = Path(segment_dir)
        if not p.is_dir():
            raise ValueError(f"{segment_dir} is not a directory")
        fn = list(p.glob("*.json"))
        if len(fn) != 1:
            m  = f"should be one json file in {segment_dir}, found {fn}"
            raise ValueError(m)
        json_filename = fn[0]
    else:
        raise ValueError("either segment_dir or json_filename must be provided")
    with open(json_filename, 'r') as f:
        metadata = json.load(f)
    return metadata

def audio_filename_to_metadata(audio_filename, tag = None):
    d, exists = utils.make_output_directory_name(audio_filename, tag)
    if not exists:
        raise ValueError(f"No segment directory found for {audio_filename}")
    metadata = segment_dir_to_metadata(d)
    return metadata

def metadata_to_audio_filename(metadata):
    return metadata[0]['audio_filename']

def directory_to_end_sample(segment_directory):
    '''find the end_sample of the last segment in a given segment_directory
    '''
    d = segment_dir_to_metadata(segment_directory)
    if not d or 'segments' not in d:
        raise ValueError(f"No segments found in {segment_directory}")
        
    end_sample = d['end_sample']
    if end_sample <= 0:
        raise ValueError(f"Invalid end samples in {segment_directory}")
    return end_sample
    
def audio_filename_to_end_sample(audio_filename):
    '''find the end_sample of the last segment for the last segment directory
    for a given audio_filename
    '''
    directories = utils.audio_filename_to_segment_directories(audio_filename)
    last_end_sample = 0
    last_end_sample_directory = None
    for directory in directories:
        try:
            end_sample = directory_to_end_sample(directory)
            if end_sample > last_end_sample:
                last_end_sample = end_sample
                last_end_sample_directory = directory
        except ValueError as e:
            print(e)
            continue
    end_sample, segment_directory = last_end_sample, last_end_sample_directory
    return end_sample, segment_directory


def handle_start_end(start_time = None, end_time = None, audio_filename = None, 
    total_n_samples = None, y = None, sample_rate = 16000,
    tag = None, overwrite = None, ignore_existing_directories = False):
    # handle total n samples
    if y is None and total_n_samples is None:
        if audio_filename is None:
            raise ValueError("either y, total_n_samples or audio_filename "
                "must be provided")
        y, _ = audio.load_audio(audio_filename, sr=sample_rate)
        total_n_samples = len(y)
    # handle start time
    if audio_filename and ignore_existing_directories is False:
        segment_directories = utils.audio_filename_to_segment_directories(
            audio_filename, ordered = True)
        if segment_directories and start_time is None:
            last_end_sample, _ = audio_filename_to_end_sample(
                audio_filename)
                
            start_sample = last_end_sample
            start_time = start_sample / sample_rate
        elif overwrite and len(segment_directories) == 1:
            name, _= utils.make_output_directory_name(audio_filename, tag)
            if name == segment_directories[0]:
                start_sample = int(start_time * sample_rate) 
        elif segment_directories and start_time is not None:
            m = f"start_time provided but segment directories exist for "
            m += f"{audio_filename}, please set start_time=None to "
            m += f"continue from the end of the last segment"
            m += f" or set ignore_existing_directories=True to ignore "
            raise ValueError(m)
    if start_time is None:
        start_time = 0.0
        start_sample = 0
    else: start_sample = int(start_time * sample_rate)
    # handle end time
    if end_time is None:
        end_time = total_n_samples / sample_rate
        final_sample = total_n_samples
    else: 
        final_sample = int(end_time * sample_rate)
        if final_sample > total_n_samples:
            print(f"end_time_seconds {end_time} is longer than the "
                "audio file, setting to end of file")
            final_sample = total_n_samples
    return start_time, end_time, start_sample, final_sample


def save_metadata(segments, output_dir):
    segments = copy.deepcopy(segments)
    for segment in segments:
        del segment['audio_segment'] 
    d = {'segments':segments, 'output_dir': str(output_dir)}
    audio_filename = segments[0]["audio_filename"]
    d['audio_filename'] = audio_filename
    tag = segments[0]['tag']
    d['tag'] = tag 
    n_segments = len(segments)
    d['n_segments'] = n_segments
    d['start_sample'] = segments[0]['start_sample']
    d['end_sample'] = segments[-1]['end_sample']
    d['n_samples'] = segments[-1]['end_sample'] - segments[0]['start_sample']
    d['sample_rate'] = segments[0]['sample_rate']
    d['start_time'] = segments[0]['start_time']
    d['end_time'] = segments[-1]['end_time']
    d['duration'] = segments[-1]['end_time'] - segments[0]['start_time']
    x = int(round(sum([s["duration"] for s in segments]) / n_segments))
    d['segment_duration'] = x
    for directory in [metadata_dir, output_dir]:
        jf = Path(directory) / f"{Path(audio_filename).stem}_{tag}"
        jf = str(jf) + f"_segments-{n_segments}_duration-{x}.json"
        json_filename = jf
        with open(json_filename, "w") as f:
            json.dump(d, f, indent=4)
        print(f"Saved segments metadata to {json_filename}")
    return d

def audio_filename_to_ordered_segment_directories(audio_filename):
    segment_directories = utils.audio_filename_to_segment_directories(
        audio_filename)
    ordered = order_segment_directories(segment_directories)
    return ordered

def order_segment_directories(segment_directories):
    filenames = [str(x).split('_')[0] for x in segment_directories]
    if len(set(filenames)) != 1:
        raise ValueError("segment directories must have the same audio filename")
    f = directory_to_end_sample
    output = sorted(segment_directories, key=lambda x: f(x))
    return output
