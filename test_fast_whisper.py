from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio
from concurrent.futures import ThreadPoolExecutor, as_completed
import builtins as __builtin__
import time
import numpy as np

def print___(*args, **kwargs):
    # __builtin__.print('New print function')
    return __builtin__.print(time.strftime("%Y-%m-%d %H:%M:%S -----  ", time.localtime()) ,*args, **kwargs)

model_id = "large-v3"
#audio_path = "./hotwords.mp3"
audio_path = "./Recording.mp3"
audio_path = "120minutes-french-TED.mp3"
audio_path = "13minutes-french-TED.mp3"
audio_path = "120min-zh-jade-bull.mp3"

# - each audio will be splitted into multiple chunks using VAD
# - adjacent chunks are merged into a segment near 30 seconds
# - `batch_size` number of segments are batched to CTranslate2 to infer results

model_cnt=4
batch_size = 64 
num_audios = 8

models = []
for x in range(model_cnt):
    model1 = WhisperModel(model_id, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model1)
    models.append(batched_model)


audio = decode_audio(audio_path, sampling_rate=model1.feature_extractor.sampling_rate)

def run_single_batch(batched_model):
    # there seems to be error in output when batch_size > 98
    single_time = time.time()
    segments, info = batched_model.transcribe(audio, batch_size=batch_size, beam_size=1)
    #print("single cost:{} index:{}".format(time.time()-single_time, index))    
    segments = list(segments) # yield all results
    actual_time = time.time()-single_time
    return segments, actual_time

def show_segments(segments, actual_time):
    def stime(sec):
        sec = int(sec)
        return f"{sec//3600:2}:{(sec//60)%60:2}:{sec%60:2}"

    max_dt = 0
    for segment in segments:
        dt = segment.end - segment.start
        if max_dt < dt: max_dt = dt
        print(f"\t {dt:.2f} [{stime(segment.start)} -> {stime(segment.end)}] {segment.text[:80]}...")
    print(f"{len(segments)} segments cost:{actual_time:.2f}  max segment dt: {max_dt:.2f} sec   {len(segments)/actual_time:.2} segments/second") 


#预热
for x in range(model_cnt):
    segments, actual_time = run_single_batch(models[x])
    show_segments(segments, actual_time)

print("test start==========")

start_time = time.time()
with ThreadPoolExecutor(max_workers=model_cnt) as t:
    futures = []
    for x in range(0, num_audios, model_cnt):
        for m in models:
            futures.append(t.submit(run_single_batch, m))

    num_segments = 0
    for future in as_completed(futures):
        segments, actual_time = future.result()
        num_segments += len(segments)

total_time_cost = time.time() - start_time
print(f"total_time_cost: {total_time_cost:.2f} sec")
print(f"               : {num_segments/total_time_cost:.2f} segments / sec")
