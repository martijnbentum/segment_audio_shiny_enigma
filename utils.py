import audio
import json
from pathlib import Path

def handle_output_directory(audio_filename, tag = None, overwrite=False):
    '''create (or overwrite) an output directory for segments
    '''
    output_dir, exists = make_output_directory_name(audio_filename, tag)
    if not exists:
        print(f"Creating output directory {output_dir}")
        output_dir.mkdir()
        return output_dir
    if not overwrite:
        m = f'Output directory {output_dir} exists, '
        m += 'use overwrite=True to overwrite'
        raise ValueError(m)
    overwrite_directory_non_recursive(output_dir)
    return output_dir
    
def make_output_directory_name(audio_filename, tag = None):
    '''make an segmeent output directory name based on the 
    audio filename and optional tag'''
    p = Path(audio_filename)
    if tag and '_' in tag:
        raise ValueError(f"tag cannot contain underscores, found: {tag}")
    directory_name = p.stem + "_segments"
    if tag is not None: directory_name += f"_{tag}"
    output_dir = Path(directory_name)
    return output_dir, directory_exists(output_dir)


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

def order_segment_directories(segment_directories):
    output = []
    f = segment_dir_to_metadata
    f = end_sample_last_segment
    output = sorted(segment_directories, key=lambda d: f(d))
    return output

def audio_filename_to_segment_directories(audio_filename, ordered = True):
    '''find all segment directories for a given audio filename
    '''
    p = Path(audio_filename)
    parent = p.parent
    stem = p.stem
    segment_dirs = list(parent.glob(stem + "_segments*"))
    return segment_dirs

def find_segment_directory(audio_filename, tag = None):
    '''find the segment directory for a given audio filename and optional tag
    if multiple segment directories exist for the audio filename and no tag is
    provided, raise an error
    '''
    segment_dirs = audio_filename_to_segment_directories(audio_filename)
    if len(segment_dirs) == 0:
        raise ValueError(f"No segment directories found for {audio_filename}")
    if tag is None:
        if len(segment_dirs) > 1:
            m = f"Multiple segment directories found for {audio_filename}: "
            m += f"{segment_dirs}, please specify a tag"
            raise ValueError(m)
        return segment_dirs[0]
    matching = [d for d in segment_dirs if d.name.endswith(tag)]
    if len(matching) == 0:
        m = f"No segment directories found for {audio_filename} with tag {tag}"
        raise ValueError(m)
    if len(matching) > 1:
        m = f"Multiple segment directories found for {audio_filename} "
        m += f"with tag {tag}: "
        m += f"{matching}, please specify a more specific tag"
        raise ValueError(m)
    return matching[0]
    
def end_sample_last_segment(segment_directory):
    '''find the end_sample of the last segment in a given segment_directory
    '''
    d = segment_dir_to_metadata(segment_directory)
    if len(d) == 0:
        raise ValueError(f"No segments found in {segment_directory}")
    end_samples = [s["end_sample"] for s in d]
    if max(end_samples) != end_samples[-1]:
        m = f"Segments in {segment_directory} are not sorted by end_samples"
        raise ValueError(m)
    if max(end_samples) <= 0:
        raise ValueError(f"Invalid end samples in {segment_directory}")
    if len(end_samples) != len(set(end_samples)):
        raise ValueError(f"Duplicate end samples in {segment_directory}")
    return max(end_samples)
    
def last_segment_audio_filename(audio_filename):
    '''find the end_sample of the last segment for the last segment directory
    for a given audio_filename
    '''
    directories = audio_filename_to_segment_directories(audio_filename)
    last_end_sample = 0
    last_end_sample_directory = None
    for directory in directories:
        try:
            end_sample = end_sample_last_segment(directory)
            if end_sample > last_end_sample:
                last_end_sample = end_sample
                last_end_sample_directory = directory
        except ValueError as e:
            print(e)
            continue
    end_sample, segment_directory = last_end_sample, last_end_sample_directory
    return end_sample, segment_directory
        
def directory_exists(directory):
    p = Path(directory)
    return p.exists() and p.is_dir()

def overwrite_directory_non_recursive(directory):
    ''' empties a directory of files, but refuses to delete anything
    if there are subdirectories, to avoid accidental data loss
    '''
    p = Path(directory)
    if not p.is_dir(): 
        raise ValueError(f"{directory} is not a directory")
    if not directory_exists(directory): return
    for item in p.iterdir():
        if item.is_dir():
            m = f'Directory {directory} contains subdirectory {item},' 
            m += f'refusing to delete non-recursively'
            raise ValueError(m)
    print(f"Overwriting contents of directory {directory}")
    for item in p.iterdir():
        item.unlink()

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
        segment_directories = audio_filename_to_segment_directories(
            audio_filename, ordered = True)
        if segment_directories and start_time is None:
            last_end_sample, _ = last_segment_audio_filename(audio_filename)
                
            start_sample = last_end_sample
            start_time = start_sample / sample_rate
        elif overwrite and len(segment_directories) == 1:
            name, _= make_output_directory_name(audio_filename, tag)
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
        

def remove_segment_directories_of_audio_filename(audio_filename):
    segment_dirs = audio_filename_to_segment_directories(audio_filename)
    for directory in segment_dirs:
        overwrite_directory_non_recursive(directory)
        print(f"Removing directory {directory}")
        directory.rmdir()
            
        
        
    
        
