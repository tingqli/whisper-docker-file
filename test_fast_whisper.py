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

cpu_threads = 32
model1 = WhisperModel(model_id, device="cuda", compute_type="float16", cpu_threads=cpu_threads)
batched_model = BatchedInferencePipeline(model=model1)

num_repeats = 1

audio = decode_audio(audio_path, sampling_rate=model1.feature_extractor.sampling_rate)
audio = np.tile(audio, num_repeats) # repeat x8


batch_size = 128
for index in range(2):
    print("==========", index)
    single_time = time.time()
    segments, info = batched_model.transcribe(audio, batch_size=batch_size, beam_size=1)
    #print("single cost:{} index:{}".format(time.time()-single_time, index))    
    segments = list(segments) # yield all results
    actual_time = time.time()-single_time

    max_dt = 0
    for segment in segments:
        dt = segment.end - segment.start
        if max_dt < dt: max_dt = dt
        print(f"\t {dt:.2f} [{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text[:80]}...")
    print(f"{len(segments)} segments cost:{actual_time:.2f}  max segment dt: {max_dt:.2f} sec   {len(segments)/actual_time:.2} segments/second") 


'''
batch_size = 12
count = 32
model_cnt = 4
cpu_threads = 12

models = []
for x in range(0, model_cnt):
    model1 = WhisperModel(model_id, device="cpu", compute_type="float32", cpu_threads=cpu_threads)
    batched_model1 = BatchedInferencePipeline(model=model1)
    models.append(batched_model1)

def run_single(batched_model, index):
    single_time = time.time()
    print("run index:", str(index))    
    segments, info = batched_model.transcribe(audio_path, batch_size=batch_size, beam_size=5)
    print("single cost:{} index:{}".format(time.time()-single_time, index))    
    for segment in segments:
        segment_time = time.time()
        print("start segment info")
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        print("segment.text cost:{} index:{}".format(time.time()-segment_time, index))
    print("single segment cost:{} index:{}".format(time.time()-single_time, index)) 

#预热
for x in range(0, model_cnt):
    run_single(models[x], x)

print("test start==========")

start_time = time.time()

with ThreadPoolExecutor(max_workers=model_cnt) as t:
    futures = []
    for x in range(0, count):
        index = x % model_cnt
        future = t.submit(run_single, models[index], x)
        futures.append(future)
    for future in as_completed(futures):
        future.result()

end_time = time.time()
print("cost=", end_time-start_time)
'''