import audio
import json
from pathlib import Path

def handle_output_directory(audio_filename, overwrite=False):
    '''create (or overwrite) an output directory for segments
    '''
    output_dir, exists = make_output_directory_name(audio_filename)
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
    
def make_output_directory_name(audio_filename):
    '''make an segmeent output directory name based on the 
    audio filename and optional tag'''
    p = Path(audio_filename)
    directory_name = p.stem + "_segments"
    output_dir = Path(directory_name)
    return output_dir, directory_exists(output_dir)

def find_segment_directory(audio_filename):
    '''find the segment directory for a given audio filename 
    '''
    p = Path(audio_filename)
    parent = p.parent
    stem = p.stem
    segment_dir = parent / f'{stem}_segments'
    if segment_dir.exists() and segment_dir.is_dir():
        return segment_dir
        
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

        

def remove_segment_directories_of_audio_filename(audio_filename):
    segment_dirs = audio_filename_to_segment_directories(audio_filename)
    for directory in segment_dirs:
        overwrite_directory_non_recursive(directory)
        print(f"Removing directory {directory}")
        directory.rmdir()
            
        
        
    
        
