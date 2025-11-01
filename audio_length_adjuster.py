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

def adjust_audio_length(audio_array, target_duration, sample_rate):
    """
    Adjust audio array to target duration by truncating or repeating.
    
    Parameters:
    - audio_array: NumPy array with shape (channels, samples)
    - target_duration: Target duration in seconds
    - sample_rate: Sample rate in Hz
    
    Returns:
    - adjusted_array: Adjusted NumPy array
    """
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

def numpy_to_mp3(audio_array, output_file, sample_rate, channels):
    """
    Convert NumPy array to MP3 file using PyAV.
    
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
    stream = container.add_stream('mp3', rate=sample_rate)
    
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


def process_audio(input_file, output_file, target_duration):
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
    adjusted_array = adjust_audio_length(audio_array, target_duration, sample_rate)
    final_duration = adjusted_array.shape[1] / sample_rate
    print(f"Final: {final_duration:.2f}s")
    
    # Write to MP3
    numpy_to_mp3(adjusted_array, output_file, sample_rate, channels)
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
    parser.add_argument('duration', type=float, help='Target duration in seconds')
    
    args = parser.parse_args()
    
    process_audio(args.input_file, args.output_file, args.duration)
    print("Successfully completed!")
        
if __name__ == "__main__":
    main()