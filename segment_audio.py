import audio
import find_silence
import metadata
from pathlib import Path
from progressbar import progressbar
import utils 

def segment_audio(audio_filename, segment_duration = 30, 
    start_time = None, end_time = None, n_segments = None, 
    sample_rate = 16000, drop_last = True, save = False, overwrite = False,
    max_distance_silence = 60):
    '''segment an audio file into segments of approximately segment_duration
    seconds, adjusting segment boundaries to the nearest silence and
    zero crossing.
    '''
    
    # check if output directory exists do not overwrite unless specified
    if not overwrite and save:
        output_dir, exists = utils.make_output_directory_name(audio_filename)
        if exists:
            m = f'Output directory {output_dir} exists, '
            m += 'use overwrite=True to overwrite'
            raise ValueError(m)

    # find silences in the audio file and zero crossings in the silences
    segment_length = int(sample_rate * segment_duration)
    y, _ = audio.load_audio(audio_filename, sr=sample_rate)
    silences = find_silence.find_silences(signal = y, sr = sample_rate,
        add_zero_crossings = True)

    total_samples = len(y)

    #find and validate start and end times and samples
    start_time, end_time, start_sample, final_sample = metadata.handle_start_end(
        start_time=start_time, end_time=end_time, audio_filename=audio_filename,
        total_n_samples=total_samples, y=y, sample_rate=sample_rate,
        overwrite=overwrite)
    print(f'start_time: {start_time}, end_time: {end_time}, '
        f'start_sample: {start_sample}, final_sample: {final_sample}, '
        f'total_samples: {total_samples}')
        
    # generate segments
    segments = []
    index = 1
    n = int((total_samples / segment_length)) + 1
    for _ in progressbar(range(n)):
        if n_segments is not None and index > n_segments:
            print(f"Reached max number of segments: {n_segments}")
            break
        end_sample = get_end_sample_index(start_sample, segment_length, y)
        print(f'start_sample: {start_sample}, target end_sample: {end_sample}')
        if end_sample >= final_sample: 
            if drop_last:
                m = f"Reached end of the file. Dropping last segment (1)"
                m += f" if you don't want this, set drop_last=False"
                print(m)
                break
            end_sample = final_sample
        else:
            silence = find_silence.find_closest_silence_with_sample_index(
                end_sample, silences, sample_rate, max_distance_silence,
                ensure_zero_crossing=True)
            silence_zero_crossing_index = 2    
            if len(silence['zero_crossings']) < 3:
                silence_zero_crossing_index = -1
                m = f"Warning: less than 3 zero crossings in silence "
                m = f'{silence}, using last zero crossing'
            end_sample = silence['zero_crossings'][silence_zero_crossing_index]
        from_end = (final_sample - end_sample) / sample_rate
        if from_end < max_distance_silence: end_sample = final_sample

        print(f'start_sample: {start_sample}, end_sample: {end_sample}')
        print(f'segment duration: {(end_sample - start_sample) / sample_rate}')
        segment_filename = make_segment_filename(audio_filename, index, 
            start_sample, end_sample)
        segment = make_segment(y, audio_filename, segment_filename, index,
            start_sample, end_sample, sample_rate,silence = silence)
        segments.append(segment)
        if end_sample >= final_sample:
            print("Reached end of the file. Stopping (2)")
            break
        start_sample = end_sample
        index += 1
    if n_segments is not None:
        print(f"Created {len(segments)} segments, max was {n_segments}")
    else:
        print(f"Created {len(segments)} segments")
    # optionally save segments to disk
    if save:
        save_segments(segments, sample_rate, audio_filename, overwrite)
    return segments

def save_segments(segments, sr, audio_filename, overwrite = False, 
    subtype="PCM_16"):
    '''save audio segments to disk and save metadata.
    '''
    p = Path(audio_filename)
    output_dir = utils.handle_output_directory(audio_filename,
        overwrite=overwrite)
    for segment in segments:
        output_filename = output_dir / segment["segment_filename"]
        y = segment["audio_segment"]
        #sf.write(output_filename, audio, sr, subtype=subtype)
        audio.save_audio(y, output_filename, sr, subtype = subtype)
        print(f"Saved segment: {output_filename}")
    print(f"Saved {len(segments)} segments to {output_dir}")
    metadata.save_metadata(segments, output_dir)

def get_end_sample_index(start_sample, segment_length, y):
    '''get start and end sample indices for a segment,
    adjusting end sample to next zero crossing after target end
    '''
    total_samples = len(y)
    target_end = start_sample + segment_length
    if target_end >= total_samples:
        return start_sample, target_end 
    end_sample = audio.next_zero_crossing(y, target_end)
    if end_sample <= start_sample:  # safety net
        m = f"end_sample ({end_sample}) <= start_sample ({start_sample})"
        raise ValueError(m)
    return end_sample 

def make_segment(y, audio_filename, segment_filename, segment_index, 
    start_sample, end_sample, sample_rate, silence):
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
        "start_time": start_sample / sample_rate,
        "end_time": end_sample / sample_rate,
        "duration": (end_sample - start_sample) / sample_rate,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "sample_rate": sample_rate,
        "audio_segment": y[start_sample:end_sample],
        "shorter": shorter,
        "silence": silence
    }
    return segment

def make_segment_filename(audio_filename, segment_index, start_sample, 
    end_sample):
    p = Path(audio_filename)
    segment_filename = f"{p.stem}"
    segment_filename += f"_n-{segment_index}"
    segment_filename += ".wav"
    return segment_filename

