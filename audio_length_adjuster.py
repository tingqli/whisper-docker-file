import argparse
import sys
import numpy as np
import av
from pathlib import Path
from fractions import Fraction

def read_audio_to_numpy(input_file):
    """
    Read audio file and return numpy array with sample rate and channels info.
    
    Parameters:
    - input_file: Path to input audio file
    
    Returns:
    - audio_array: NumPy array with audio data (float32, range [-1.0, 1.0])
    - sample_rate: Sample rate in Hz
    - channels: Number of audio channels
    """
    try:
        container = av.open(input_file)
        audio_stream = next(s for s in container.streams if s.type == 'audio')
        
        sample_rate = audio_stream.rate
        channels = audio_stream.channels
        
        audio_frames = []
        for frame in container.decode(audio=0):
            # Convert to numpy array and normalize to float32
            audio_data = frame.to_ndarray()
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            elif audio_data.dtype == np.float64:
                audio_data = audio_data.astype(np.float32)
            
            audio_frames.append(audio_data)
        
        container.close()
        
        if not audio_frames:
            raise ValueError("No audio data found in the file")
        
        # Combine all frames
        audio_array = np.concatenate(audio_frames, axis=1)
        
        # If multi-channel, shape should be (channels, samples)
        # If single channel, ensure shape is (1, samples)
        if len(audio_array.shape) == 1:
            audio_array = audio_array.reshape(1, -1)
        
        return audio_array, sample_rate, channels
        
    except Exception as e:
        raise Exception(f"Error reading audio file: {str(e)}")

def adjust_audio_length(audio_array, start, target_duration, sample_rate):
    """
    Adjust audio array to target duration by truncating or repeating.
    
    Parameters:
    - audio_array: NumPy array with shape (channels, samples)
    - target_duration: Target duration in seconds
    - sample_rate: Sample rate in Hz
    
    Returns:
    - adjusted_array: Adjusted NumPy array
    """
    start_sample = int(start * sample_rate)
    audio_array = audio_array[:, start_sample:]
    current_samples = audio_array.shape[1]
    target_samples = int(target_duration * sample_rate)
    
    if target_samples == current_samples:
        return audio_array
    
    elif target_samples < current_samples:
        # Truncate: take the first target_samples
        print(f"Truncating audio from {current_samples/sample_rate:.2f}s to {target_duration}s")
        return audio_array[:, :target_samples]
    
    else:
        # Repeat: repeat the audio until we reach target length
        print(f"Extending audio from {current_samples/sample_rate:.2f}s to {target_duration}s by repeating")
        repeats_needed = int(np.ceil(target_samples / current_samples))
        repeated_array = np.tile(audio_array, (1, repeats_needed))
        return repeated_array[:, :target_samples]

def numpy_to_file(audio_array, output_file, sample_rate, channels):
    """
    Convert NumPy array to target file using PyAV.
    
    Parameters:
    - audio_array: NumPy array with shape (channels, samples) in float32 format
    - output_file: Output MP3 file path
    - sample_rate: Sample rate in Hz
    - channels: Number of audio channels
    """
    # Convert float32 [-1.0, 1.0] to int16
    audio_int16 = (audio_array * 32767.0).astype(np.int16)
    # Create output container
    container = av.open(output_file, mode='w')
    
    # Add MP3 stream with correct channel count
    if output_file.endswith("wav"):
        codec_name = 'pcm_s16le'
    elif output_file.endswith("mp3"):
        codec_name = 'mp3'
    else:
        assert False, f"unknown file ext: {output_file}"
        
    stream = container.add_stream(codec_name, rate=sample_rate)
    
    # Set channels through the layout (this is the correct way)
    if channels == 1:
        layout = 'mono'
    elif channels == 2:
        layout = 'stereo'
    else:
        layout = f'{channels}c'
    
    # Process in chunks to handle large files
    chunk_size = 1024  # samples per chunk
    total_samples = audio_int16.shape[1]

    for i in range(0, total_samples, chunk_size):
        end_idx = min(i + chunk_size, total_samples)
        chunk = audio_int16[:, i:end_idx]
        if channels == 1:
            chunk = chunk[:1,:]
        # Create audio frame for this chunk
        frame = av.AudioFrame.from_ndarray(chunk, format='s16p', layout=layout)
        frame.rate = sample_rate
        frame.time_base = Fraction(1, sample_rate)

        # Encode and write the frame
        for packet in stream.encode(frame):
            container.mux(packet)
    
    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()

def resample_audio(audio_array, original_rate, target_rate):
    """
    Resample audio array to target sample rate using PyAV.
    
    Parameters:
    - audio_array: NumPy array with shape (channels, samples)
    - original_rate: Original sample rate in Hz
    - target_rate: Target sample rate in Hz
    
    Returns:
    - resampled_array: Resampled NumPy array
    """
    if original_rate == target_rate:
        return audio_array

    # Convert to int16 for resampling
    audio_int16 = (audio_array * 32767.0).astype(np.int16)
    
    # Create input layout
    channels = audio_int16.shape[0]
    if channels == 1:
        layout = 'mono'
    elif channels == 2:
        layout = 'stereo'
    else:
        layout = f'{channels}c'
    
    # Create input frame
    input_frame = av.AudioFrame.from_ndarray(audio_int16, format='s16p', layout=layout)
    input_frame.rate = original_rate
    input_frame.time_base = Fraction(1, original_rate)
    
    # Create resampler
    resampler = av.audio.resampler.AudioResampler(
        format='s16p',
        layout=layout,
        rate=target_rate
    )
    
    # Resample
    resampled_frames = []
    for output_frame in resampler.resample(input_frame):
        resampled_data = output_frame.to_ndarray()
        resampled_frames.append(resampled_data)
    
    if not resampled_frames:
        raise ValueError("Resampling produced no output")
    
    # Combine resampled frames
    resampled_array = np.concatenate(resampled_frames, axis=1)
    
    # Convert back to float32
    resampled_array_float = resampled_array.astype(np.float32) / 32768.0
    
    print(f"Resampled audio from {original_rate}Hz to {target_rate}Hz")
    return resampled_array_float
    

def process_audio(input_file, output_file, start, target_duration, target_samplerate):
    """
    Main function to process audio file.
    
    Parameters:
    - input_file: Path to input audio file
    - output_file: Path to output MP3 file
    - target_duration: Target duration in seconds
    """
    # Validate inputs
    if not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if target_duration <= 0:
        raise ValueError("Target duration must be positive")
    
    print(f"Processing: {input_file}")
    print(f"Target duration: {target_duration} seconds")
    
    # Read audio file
    audio_array, sample_rate, channels = read_audio_to_numpy(input_file)

    original_duration = audio_array.shape[1] / sample_rate
    print(f"Original: {original_duration:.2f}s, {sample_rate}Hz, {channels} channel(s)")
    
    # Adjust audio length
    adjusted_array = adjust_audio_length(audio_array, start, target_duration, sample_rate)
    final_duration = adjusted_array.shape[1] / sample_rate
    print(f"Final: {final_duration:.2f}s")

    if target_samplerate >0 and target_samplerate != sample_rate:
        adjusted_array = resample_audio(adjusted_array, sample_rate, target_samplerate)
        sample_rate = target_samplerate

    # Write to output
    numpy_to_file(adjusted_array, output_file, sample_rate, channels)
    print(f"Output saved: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Adjust audio file length and convert to MP3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.wav output.mp3 10.5
  %(prog)s input.mp3 output.mp3 30
  %(prog)s "input file.flac" output.mp3 15.0
        """
    )
    
    parser.add_argument('input_file', help='Input audio file path')
    parser.add_argument('output_file', help='Output MP3 file path')
    parser.add_argument('-s', '--start', type=float, default=0.0, help='start position')
    parser.add_argument('-r', '--sr', type=int, default=0, help='target sample-rate')
    parser.add_argument('duration', type=float, help='Target duration in seconds')

    args = parser.parse_args()

    process_audio(args.input_file, args.output_file, args.start, args.duration, args.sr)
    print("Successfully completed!")
        
if __name__ == "__main__":
    main()