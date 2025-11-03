import whisper
import numpy as np
from faster_whisper.audio import decode_audio

# unlike faster-whisper which uses silero-vad to handle long audios,
# openai whisper relying on whisper's timestamps output to move overlapping
# windows along mel-spectrums and many tricks to avoid repetations & hallucinations
#
# but openai's methods has no support for batched inferences because it relies on previous 30s-window's
# result to decide where next 30s-window begins, while faster-whisper's methods can cut 
# a single long audio into many non-overlaping 30s-windows and speed-up inference by batching
# them together.
#
# vLLM has no solution for long audio and relies on user to cut audio into 30s windows.
 

model = whisper.load_model("base.en")
print(
    f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
    f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
)
# options = whisper.DecodingOptions(language="en", without_timestamps=True)
options = dict(language="en", beam_size=5, best_of=5)
transcribe_options = dict(task="transcribe", **options)

audio_path = "./Recording.mp3"
audio_path = "120minutes-french-TED.mp3"
audio_path = "120min-zh-jade-bull.mp3"
audio_path = "13minutes-french-TED.mp3"
audio_path = "1min-en-TED.mp3"

audio = decode_audio(audio_path, sampling_rate=16000)

transcription = model.transcribe(audio, **transcribe_options)["text"]

print(transcription)
