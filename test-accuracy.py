
# pip install whisper_normalizer zhconv cntn datasets==3.6.0 evaluate jiwer

import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

from datasets import load_dataset, Audio
import torch
import glob, librosa
from tqdm import tqdm
import numpy as np

with_profiler = os.environ.get("VLLM_TORCH_PROFILER_DIR", None) is not None

model_id = "whisper-large-v3"

def speech_recognition_hf(dataset):
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    from transformers.pipelines.pt_utils import KeyDataset
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device
    )

    predictions = []
    # https://huggingface.co/docs/transformers/main_classes/pipelines#pipeline-batching
    for i in tqdm(range(dataset.num_rows)):
        result = pipe(dataset[i]["audio"], generate_kwargs={"language": dataset[i]["language"]})
        predictions.append(result["text"])

    return predictions

def speech_recognition_firered_asr(dataset, beam_size = 1, batch_size = 16):
    # need to install fireredasr & download FireRedASR-AED-L
    # https://github.com/FireRedTeam/FireRedASR?tab=readme-ov-file#setup
    # huggingface-cli download --resume-download FireRedTeam/FireRedASR-AED-L --local-dir FireRedASR-AED-L
    from fireredasr.models.fireredasr import FireRedAsr
    model = FireRedAsr.from_pretrained("aed", "FireRedASR-AED-L")
    predictions = []
    batch_uttid = []
    batch_wav_path = []

    for i in tqdm(range(dataset.num_rows)):
        batch_uttid.append(i)
        batch_wav_path.append(dataset[i]["audio"]["path"])
        if len(batch_uttid) >= batch_size or i == (dataset.num_rows - 1):
            results = model.transcribe(
                batch_uttid,
                batch_wav_path,
                {
                    "use_gpu": 1,
                    "beam_size": beam_size,
                    "nbest": 1,
                    "decode_max_len": 0,
                    "softmax_smoothing": 1.0,
                    "aed_length_penalty": 0.0,
                    "eos_penalty": 1.0
                }
            )
            for r in results:
                predictions.append(r["text"])
            batch_uttid = []
            batch_wav_path = []
    return predictions


def speech_recognition_faster_whisper(dataset, beam_size = 1):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    predictions = []
    for i in tqdm(range(dataset.num_rows)):
        audio = dataset[i]["audio"]["array"]
        language = dataset[i]["language"]
        # segments, info = batched_model.transcribe(audio=audio, batch_size=8, beam_size=beam_size, without_timestamps=True, language=language)
        segments, info = model.transcribe(audio=audio, beam_size=beam_size, without_timestamps=True, language=language, language_detection_segments=0)
        # print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
        predictions.append("".join([s.text for s in segments]))
    return predictions

def speech_recognition_vllm(dataset, batch_size = 1, max_tokens = 500, use_fp8_kv_cache=False, enforce_eager=False):
    from vllm import LLM, SamplingParams    
    prompts = []
    for i in range(dataset.num_rows):
        language = dataset[i]["language"]
        chunk = (dataset[i]["audio"]["array"], dataset[i]["audio"]["sampling_rate"])
        prompts.append({
            "prompt": f"<|startoftranscript|><|{language}|><|transcribe|><|notimestamps|>",
            "multi_modal_data": {
                "audio": chunk,
            },
        })

    kwargs = {}
    if use_fp8_kv_cache:
        kwargs["kv_cache_dtype"] = "fp8"
    if enforce_eager:
        kwargs["enforce_eager"] = True
    llm = LLM(
            model=model_id,
            max_model_len=448,
            max_num_seqs=batch_size,
            limit_mm_per_prompt={"audio": 1},
            **kwargs
    )
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1.0,
        max_tokens=max_tokens, #500,
    )
    import time
    #llm.start_profile()
    if with_profiler: llm.start_profile()
    start = time.time()
    outputs = llm.generate(prompts,sampling_params)
    duration = time.time() - start
    if with_profiler: llm.stop_profile()
    #llm.stop_profile()
    print(f"len(prompts):{len(prompts)}")
    print("Duration:", duration)
    print("RPS:", len(prompts) / duration)

    return [x.outputs[0].text for x in outputs]


def load_1000_datasets():
    dataset = load_dataset("csv", data_files=os.path.join("accuracy_issue","1000case_result.csv"), split="train")
    sr = 16000
    wav_path = os.path.join("accuracy_issue","1000")
    def preprocess(row):
        audio_file = None
        audio = None
        sample_rate = 0
        text = row["segment_predict_whisper_large_v3_text"]
        if not isinstance(text, str):
            print(f">>>>>>>>> skip {row['aid']} with no expected answer")
        else:
            audio_files = glob.glob(os.path.join(wav_path, f'*{row["aid"]}*.wav'))
            assert len(audio_files) == 1
            audio_file = audio_files[0]
            audio, sample_rate = librosa.load(audio_file,sr=None, dtype=np.float32)
            if sample_rate != sr:
                audio = librosa.resample(audio.numpy().astype(np.float32), orig_sr=sample_rate, target_sr=sr)
                sample_rate = sr
            else:
                audio = audio.astype(np.float32)

            duration = audio.shape[0] / sample_rate
            if audio.shape[0] >= sample_rate*30:
                print(f">>>>>>>>> skip {audio_file} with duration {duration} > 30 ")
                sample_rate = 0
                audio = None

        row["audio_file"] = audio_file
        row["text"] = text
        row["language"] = row["segment_language"]
        row["audio"] = {"array":audio, "sampling_rate":sample_rate}
        return row

    # https://discuss.huggingface.co/t/dataset-map-return-only-list-instead-torch-tensors/15767
    dataset = dataset.add_column("audio", [None] * len(dataset))
    dataset.set_format("numpy", columns=["audio"], output_all_columns=True)
    dataset = dataset.map(preprocess)
    dataset = dataset.filter(lambda example: example["audio"]["array"] is not None and example["language"] == "en" )
    return dataset

def load_demo_datasets():
    dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", split="validation")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    def preprocess(row):
        row["language"] = "en"
        return row
    return dataset.map(preprocess)

def load_librispeech_asr_test_datasets():
    dataset = load_dataset("kresnik/librispeech_asr_test", "clean", split="test", trust_remote_code=True)
    def preprocess(row):
        row["language"] = "en"
        return row
    return dataset.map(preprocess)

def load_WenetSpeech_datasets():
    # RnpfulybhtQBASxeBjygvonADhOwSnNAUj
    #dataset = load_dataset("TwinkStart/WenetSpeech", split="test_meeting", trust_remote_code=True)
    ws_test_meeting = load_dataset("wenet-e2e/wenetspeech", "TEST_MEETING", split="test", trust_remote_code=True)
    def preprocess(row):
        row["language"] = "zh"
        return row
    return ws_test_meeting.map(preprocess)


if __name__ == "__main__":
    # https://huggingface.co/docs/datasets/access
    #dataset = load_demo_datasets()
    # 
    #dataset = load_1000_datasets()

    # HF wer_score: 1.74 %
    # vLLM batch-size1  ,kv-cache-fp16: wer_score 1.88 %
    # vLLM batch-size1  ,kv-cache-fp8 : wer_score 1.88 %
    # vLLM batch-size256,kv-cache-fp8 : wer_score 1.88 %
    # fireredasr FireRedASR-AED-L: wer_score: 97.66 
    # faster_whisper beam_size=1: wer_score: 1.79 %  5.05it/s
    # faster_whisper beam_size=5: wer_score: 1.74 %  4.52it/s
    #dataset = load_librispeech_asr_test_datasets()

    # whisper-large-v3 on wenetspeech :
    #  https://arxiv.org/pdf/2501.14350       CER:18.87
    #  https://zhuanlan.zhihu.com/p/662906303 CER:20.15
    
    # faster_whisper beam_size=1        4.90it/s:  wer_score: 97.62 % cer_score: 21.37 %
    # faster_whisper beam_size=5        4.43it/s:  wer_score: 96.67 % cer_score: 19.36 %
    # fireredasr FireRedASR-AED-L  H20: 4.86it/s:  wer_score: 58.77 % cer_score: 4.96 %
    # fireredasr FireRedASR-AED-L  308:
    # vLLM batch256,kv-cache-fp8 H20 RPS: 31.10 wer_score: 97.50 % cer_score: 20.40 %
    # vLLM batch256,kv-cache-fp8 308 RPS: 31.67: wer_score: 97.61 %  cer_score: 20.52 %
    dataset = load_WenetSpeech_datasets()
    dataset = dataset.filter(lambda example: example["audio"]["array"].shape[0] < 30*16000)
    dataset = dataset.select(range(0,20))

    print(dataset)

    #results = speech_recognition_hf(dataset)
    #results = speech_recognition_vllm(dataset, use_fp8_kv_cache=True, batch_size=256)
    #results = speech_recognition_faster_whisper(dataset, beam_size=1)
    results = speech_recognition_firered_asr(dataset, beam_size=1)

    # https://pypi.org/project/whisper-normalizer/
    from whisper_normalizer.english import EnglishTextNormalizer
    from whisper_normalizer.basic import BasicTextNormalizer
    from zhconv import convert
    # https://github.com/open-speech/cn-text-normalizer
    import cntn
    def ZhTextNormalizer(x):
        x = convert(x, "zh-cn")
        x = cntn.w2s(x)
        return x

    text_normalizers = {
        "zh":ZhTextNormalizer,
        "en":EnglishTextNormalizer(),
        "basic" : BasicTextNormalizer()
    }

    predictions = []
    references = []
    for row, pred in zip(dataset, results):
        language_code = row["language"]
        if language_code not in text_normalizers: language_code = "basic"
        norm = text_normalizers[language_code]
        pred = norm(pred)
        ref = norm(row["text"])
        predictions.append(pred)
        references.append(ref)

        if 1:
            print("========== ", row["audio"]["path"])
            print(ref)
            print(pred)

    # print both WER & CER
    # https://huggingface.co/learn/audio-course/zh-CN/chapter5/evaluation
    import evaluate
    wer = evaluate.load("wer")
    cer = evaluate.load("cer")
    print(f"wer_score: {wer.compute(predictions=predictions, references=references) * 100:.2f} %")
    print(f"cer_score: {cer.compute(predictions=predictions, references=references) * 100:.2f} %")
