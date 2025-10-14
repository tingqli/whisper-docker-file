import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

import time
import requests
from datasets import Audio
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from librosa import resample, load
# Create a Whisper encoder/decoder model instance
sr = 16000
batch_size = 256
num_prompts = batch_size*5
max_tokens = 500

if 0:
    batch_size = 256
    num_prompts = batch_size*2
    max_tokens = 16

"""
init vllm
"""
llm = LLM(
        model="whisper-large-v3",
        max_model_len=448,
        max_num_seqs=batch_size,
        limit_mm_per_prompt={"audio": 1},
        # dtype="float16",
        kv_cache_dtype="fp8",
        #enforce_eager=True
)
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    top_p=1.0,
    max_tokens=max_tokens, #500,
    #max_tokens=0
)


"""
warm up
"""

EXPECTED = {
    "openai/whisper-tiny": [
        " He has birth words I spoke in the original corner of that. And a"
        " little piece of black coat poetry. Mary had a little sandwich,"
        " sweet, with white and snow. And everyone had it very went the last"
        " would sure to go.",
        " >> And the old one, fit John the way to Edgar Martinez. >> One more"
        " to line down the field line for our base camp. Here comes joy. Here"
        " is June and the third base. They're going to wave him in. The throw"
        " to the plate will be late. The Mariners are going to play for the"
        " American League Championship. I don't believe it. It just continues"
        " by all five.",
    ],
    "openai/whisper-small": [
        " The first words I spoke in the original pornograph. A little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite a"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the old one pitch on the way to Edgar Martinez one month. Here"
        " comes joy. Here is Junior to third base. They're gonna wave him"
        " in. The throw to the plate will be late. The Mariners are going to"
        " play for the American League Championship. I don't believe it. It"
        " just continues. My, oh my.",
    ],
    "openai/whisper-medium": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its fleece was quite as"
        " slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez swung on the line"
        " down the left field line for Obeyshev. Here comes Joy. Here is"
        " Jorgen at third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh"
        " my.",
    ],
    "openai/whisper-large-v3": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its feet were quite as"
        " slow, and everywhere that Mary went, the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line."
        " Now the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my.",
    ],
    "openai/whisper-large-v3-turbo": [
        " The first words I spoke in the original phonograph, a little piece"
        " of practical poetry. Mary had a little lamb, its streets were quite"
        " as slow, and everywhere that Mary went the lamb was sure to go.",
        " And the 0-1 pitch on the way to Edgar Martinez. Swung on the line"
        " down the left field line for a base hit. Here comes Joy. Here is"
        " Junior to third base. They're going to wave him in. The throw to the"
        " plate will be late. The Mariners are going to play for the American"
        " League Championship. I don't believe it. It just continues. My, oh,"
        " my.",
    ],
}

warmup_prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
    #"decoder_prompt": "<|startoftranscript|>",
    {  # Test explicit encoder/decoder prompt
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": AudioAsset("winning_call").audio_and_sample_rate,
            },
        },
        "decoder_prompt": "<|startoftranscript|>",
    }]

outputs  = llm.generate(warmup_prompts,sampling_params)
for output, expected in zip(outputs, EXPECTED["openai/whisper-large-v3"]):
    print(output.outputs[0].text)
    assert output.outputs[0].text == expected


print("[INFO] warm up ok")
"""
load audio
"""

audio_file = "out.wav"
audio, sample_rate = load(audio_file,sr=None)
if sample_rate != sr:
    # Use librosa to resample the audio
    audio = resample(audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=16000)
    print(f"File: {file}, Sample rate: {sample_rate}, Audio shape: {audio.shape}, Duration: {audio.shape[0] / sample_rate:.2f} seconds")
chunk = (audio, sr)


prompts = [
    {
        "prompt": "<|startoftranscript|>",
        "multi_modal_data": {
            "audio": chunk,#AudioAsset("mary_had_lamb").audio_and_sample_rate,
        },
    },
]*num_prompts

 
print("[INFO] test begin")
 
#llm.start_profile()
start = time.time()
outputs = llm.generate(prompts,sampling_params)
duration = time.time() - start
#llm.stop_profile()


print("[INFO] test end")
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    #encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    #print(f"Generated text: {generated_text!r}")
 

print(f"len(prompts):{len(prompts)}")
print("Duration:", duration)
print("RPS:", len(prompts) / duration)
print(generated_text)

 
