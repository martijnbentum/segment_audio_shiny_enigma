import audio
import copy
import json
import utils
from pathlib import Path

metadata_dir = Path('metadata/')

def audio_filename_to_summary(audio_filename):
    '''print a summary of segment metadata for a given audio filename
    '''
    segment_directory = utils.find_segment_directory(audio_filename)
    end_sample, _ = audio_filename_to_end_sample(audio_filename)
    metadatas = []
    total_duration = 0.0
    segment_durations = []
    n_segments = []
    print('\n' + '-' * 40)
    print(f"Summary for: {audio_filename}")
    print(f"Found segment directory {segment_directory}") 
    print('\n' + '-' * 40)
    metadata = audio_filename_to_metadata(audio_filename)
    segment_durations = metadata['segment_durations']
    n_segments = metadata['n_segments']
    total_duration = metadata['duration']
    print("-" * 40)
    m = f"Audio filename: {audio_filename}\n"
    m += f"Number of segments: {n_segments}\n"
    m += f"Segment durations: {', '.join(map(str,map(int,segment_durations)))}\n"
    m += f"avg segment duration: {metadata['average_segment_duration']} seconds\n"
    m += f"Total duration: {total_duration:.3f} seconds\n"
    m += f"End sample of last segment: {end_sample}\n"
    end_time = end_sample / metadata['sample_rate']
    m += f"End time of last segment: {end_time:.3f} seconds\n"
    print(m)
    
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

def audio_filename_to_metadata(audio_filename) :
    '''load segment metadata for a given audio filename
    '''
    d, exists = utils.make_output_directory_name(audio_filename)
    if not exists:
        raise ValueError(f"No segment directory found for {audio_filename}")
    metadata = segment_dir_to_metadata(d)
    return metadata

def metadata_to_audio_filename(metadata):
    '''get the audio filename from a metadata file.
    '''
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
    
def handle_start_end(start_time = None, end_time = None, audio_filename = None, 
    total_n_samples = None, y = None, sample_rate = 16000,
    overwrite = None):
    '''sets and validates start and end times and samples for an audio file'''
    # handle total n samples
    if y is None and total_n_samples is None:
        if audio_filename is None:
            raise ValueError("either y, total_n_samples or audio_filename "
                "must be provided")
        y, _ = audio.load_audio(audio_filename, sr=sample_rate)
        total_n_samples = len(y)
    # handle start time
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
    '''save segment metadata to a json file in the metadata directory
    and the segment output directory.
    '''
    segments = copy.deepcopy(segments)
    for segment in segments:
        del segment['audio_segment'] 
    d = {'segments':segments, 'output_dir': str(output_dir)}
    audio_filename = segments[0]["audio_filename"]
    d['audio_filename'] = audio_filename
    n_segments = len(segments)
    d['n_segments'] = n_segments
    d['start_sample'] = segments[0]['start_sample']
    d['end_sample'] = segments[-1]['end_sample']
    d['n_samples'] = segments[-1]['end_sample'] - segments[0]['start_sample']
    d['sample_rate'] = segments[0]['sample_rate']
    d['start_time'] = segments[0]['start_time']
    d['end_time'] = segments[-1]['end_time']
    d['duration'] = segments[-1]['end_time'] - segments[0]['start_time']
    durations = [s['duration'] for s in segments]
    x = int(round(sum(durations) / n_segments))
    d['average_segment_duration'] = x
    d['segment_durations'] = durations
    od = f'{output_dir}/' 
    d['segment_filenames'] = [od + s['segment_filename'] for s in segments]
    for directory in [metadata_dir, output_dir]:
        jf = Path(directory) / f"{Path(audio_filename).stem}.json"
        json_filename = jf
        with open(json_filename, "w") as f:
            json.dump(d, f, indent=4)
        print(f"Saved segments metadata to {json_filename}")
    return d

def audio_filename_to_metadata_info(audio_filename):
    '''get metadata info (without segments) for a given audio filename
    '''
    d = audio_filename_to_metadata(audio_filename)
    del d['segments']
    return d

def audio_filename_to_metadata(audio_filename):
    '''load the metadata for a given audio filename.
    '''
    f = get_metadata_filename(audio_filename)
    if f is None:
        print(f"No metadata file found for {audio_filename}")
        return None
    metadata = load_metadata(f)
    return metadata

def get_metadata_filename(audio_filename):
    '''get the metadata filename for a given audio filename.
    '''
    filename = metadata_dir / f"{Path(audio_filename).stem}.json"
    if filename.is_file():
        return filename

def load_metadata(json_filename):
    '''load metadata from a given json filename.
    '''
    print(f"Loading metadata from {json_filename}")
    with open(json_filename, 'r') as f:
        metadata = json.load(f)
    return metadata
