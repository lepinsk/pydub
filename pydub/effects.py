import sys
from .utils import (
    db_to_float,
    ratio_to_db,
    register_pydub_effect,
    make_chunks,
    audioop,
)
from .exceptions import TooManyMissingFrames
from tempfile import NamedTemporaryFile
import subprocess
import os

if sys.version_info >= (3, 0):
    xrange = range

@register_pydub_effect
def noise_reduction(seg, sample_start=0, sample_end=1000, sensitivity=0.2):
    """
    noise_reduction will remove consistent background noise from a recording
        given a sample region which is expected to be silent (but includes the
        background noise)

    sample_start: the time (in ms) at which we should start the sample period
    sample_end: the time (in ms) at which we end the sample period
    sensitivity: (from 0 to 1) how aggresive to be with the noise reduction
    """

    sample_section = seg[sample_start:sample_end]
    sample_section_file = NamedTemporaryFile(mode='wb', delete=False)
    sample_section.export(sample_section_file.name, format="wav")
    
    noise_profile_file = NamedTemporaryFile(mode='wb', delete=False)
    noise_profile_command = ['sox', sample_section_file.name, '-n', 'noiseprof', noise_profile_file.name]
    subprocess.call(noise_profile_command, stderr=open(os.devnull))

    noisy_file = NamedTemporaryFile(mode='wb', suffix=".wav", delete=False)
    seg.export(noisy_file.name, format="wav")
    denoised_file = NamedTemporaryFile(mode='wb', suffix=".wav", delete=False)
    noise_reduction_command = ['sox', noisy_file.name, denoised_file.name, 'noisered', noise_profile_file.name, '0.2']
    subprocess.call(noise_reduction_command, stderr=open(os.devnull))

    os.remove(sample_section_file.name)
    os.remove(noise_profile_file.name)
    os.remove(noisy_file.name)

    denoised_segment = seg.from_file(denoised_file.name)
    
    os.remove(denoised_file.name)

    return denoised_segment

@register_pydub_effect
def sox_compand(seg):
    """
    right now a very simple wrapper around sox's compand

    no params for the time being...
    """
    uncompressed_file = NamedTemporaryFile(mode='wb', suffix=".wav", delete=False)
    seg.export(uncompressed_file.name, format="wav")

    compressed_file = NamedTemporaryFile(mode='wb', suffix=".wav", delete=False)
    compression_command = ['sox', uncompressed_file.name, compressed_file.name, "compand", "0.3,1", "6:-70,-60,-20", "-5", "-90", "0.2"];
    subprocess.call(compression_command, stderr=open(os.devnull))

    os.remove(uncompressed_file.name)

    compressed_segment = seg.from_file(compressed_file.name)

    os.remove(compressed_file.name)

    return compressed_segment

@register_pydub_effect
def normalize(seg, headroom=0.1):
    """
    headroom is how close to the maximum volume to boost the signal up to (specified in dB)
    """
    peak_sample_val = seg.max
    
    # if the max is 0, this audio segment is silent, and can't be normalized
    if peak_sample_val == 0:
        return seg
    
    target_peak = seg.max_possible_amplitude * db_to_float(-headroom)

    needed_boost = ratio_to_db(target_peak / peak_sample_val)
    return seg.apply_gain(needed_boost)


@register_pydub_effect
def speedup(seg, playback_speed=1.5, chunk_size=150, crossfade=25):
    # we will keep audio in 150ms chunks since one waveform at 20Hz is 50ms long
    # (20 Hz is the lowest frequency audible to humans)

    # portion of AUDIO TO KEEP. if playback speed is 1.25 we keep 80% (0.8) and
    # discard 20% (0.2)
    atk = 1.0 / playback_speed

    if playback_speed < 2.0:
        # throwing out more than half the audio - keep 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size * (1 - atk) / atk)
    else:
        # throwing out less than half the audio - throw out 50ms chunks
        ms_to_remove_per_chunk = int(chunk_size)
        chunk_size = int(atk * chunk_size / (1 - atk))

    # the crossfade cannot be longer than the amount of audio we're removing
    crossfade = min(crossfade, ms_to_remove_per_chunk - 1)

    # DEBUG
    #print("chunk: {0}, rm: {1}".format(chunk_size, ms_to_remove_per_chunk))

    chunks = make_chunks(seg, chunk_size + ms_to_remove_per_chunk)
    if len(chunks) < 2:
        raise Exception("Could not speed up AudioSegment, it was too short {2:0.2f}s for the current settings:\n{0}ms chunks at {1:0.1f}x speedup".format(
            chunk_size, playback_speed, seg.duration_seconds))

    # we'll actually truncate a bit less than we calculated to make up for the
    # crossfade between chunks
    ms_to_remove_per_chunk -= crossfade

    # we don't want to truncate the last chunk since it is not guaranteed to be
    # the full chunk length
    last_chunk = chunks[-1]
    chunks = [chunk[:-ms_to_remove_per_chunk] for chunk in chunks[:-1]]

    out = chunks[0]
    for chunk in chunks[1:]:
        out = out.append(chunk, crossfade=crossfade)

    out += last_chunk
    return out
    
@register_pydub_effect
def strip_silence(seg, silence_len=1000, silence_thresh=-20):
    silence_thresh = seg.rms * db_to_float(silence_thresh)
    
    # find silence and add start and end indicies to the to_cut list
    to_cut = []
    silence_start = None
    for i, sample in enumerate(seg):
        if sample.rms < silence_thresh:
            if silence_start is None:
                silence_start = i
            continue
            
        if silence_start is None:
            continue
            
        if i - silence_start > silence_len:
            to_cut.append([silence_start, i-1])
        
        silence_start = None
            
    # print(to_cut)
    
    keep_silence = 100
    
    to_cut.reverse()
    for cstart, cend in to_cut:
        if len(seg[cend:]) < keep_silence:
            seg = seg[:cstart + keep_silence]
        elif len(seg[:cstart]) < keep_silence:
            seg = seg[cend-keep_silence:]
        else:
            #print(cstart, "-", cend)
            seg = seg[:cstart+keep_silence].append(seg[cend-keep_silence:], crossfade=keep_silence*2)
    return seg

@register_pydub_effect
def noise_gate(seg, silence_len=1000, silence_thresh=-20, crossfade_time=500):
    silence_thresh = seg.rms * db_to_float(silence_thresh)
    seg_duration_ms = seg.duration_seconds*1000

    # find silence and add start and end indicies to the to_cut list
    to_cut = []
    silence_start = None
    for i, sample in enumerate(seg):
        if sample.rms < silence_thresh:
            if silence_start is None:
                silence_start = i
            continue
            
        if silence_start is None:
            continue
            
        if i - silence_start > silence_len:
            to_cut.append([silence_start, i-1])
        
        silence_start = None
    
    crossfade_buffer = crossfade_time / 2

    #combine gate regions separated by less than our crossfade length
    i = 0
    while i < (len(to_cut) - 1):
        cstart = to_cut[i][0]
        cend = to_cut[i][1]
        cstart_next = to_cut[i+1][0]
        cend_next = to_cut[i+1][1]
        if (cstart_next - cend) < crossfade_time:
            to_cut.pop(i+1)
            to_cut.pop(i)
            insert_array = [cstart, cend_next]
            to_cut.insert(i, insert_array)
        i = i+1

    for cstart, cend in to_cut:
        if cstart == 0 or (cstart - crossfade_buffer) < 0:                                              #start cut
            cstart = 0
            seg_front = seg[:cend + crossfade_buffer] - 120
            seg_back = seg[cend - crossfade_buffer:]
            seg = seg_front.append(seg_back, crossfade=crossfade_time)
        elif cend == (seg.duration_seconds * 1000) or (cend + crossfade_buffer) > seg_duration_ms:      #end cut
            cend = seg_duration_ms
            seg_front = seg[:cstart + crossfade_buffer]
            seg_back = seg[cstart - crossfade_buffer:] - 120
            seg = seg_front.append(seg_back, crossfade=crossfade_time)
        else:                                                                                           #middle cut
            seg_front = seg[:cstart + crossfade_buffer]
            seg_middle = seg[cstart - crossfade_buffer:cend + crossfade_buffer] - 120
            seg_back = seg[cend - crossfade_buffer:]
            seg = seg_front.append(seg_middle, crossfade=crossfade_time).append(seg_back, crossfade=crossfade_time)

    return seg

@register_pydub_effect
def compress_dynamic_range(seg, threshold=-20.0, ratio=4.0, attack=5.0, release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Compression ratio. Audio louder than the threshold will be 
            reduced to 1/ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    
    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms
    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    attenuation = 0.0
    
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in xrange(int(seg.frame_count())):
        rms_now = rms_at(i)
        
        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_attenuation = (1 - (1.0 / ratio)) * db_over_threshold(rms_now)
        
        attenuation_inc = max_attenuation / attack_frames
        attenuation_dec = max_attenuation / release_frames
        
        if rms_now > thresh_rms and attenuation <= max_attenuation:
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else:
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)
        
        frame = seg.get_frame(i)
        if attenuation != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))
        
        output.append(frame)
    
    return seg._spawn(data=b''.join(output))
