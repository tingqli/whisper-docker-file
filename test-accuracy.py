import os
os.environ['VLLM_ATTENTION_BACKEND'] = 'XFORMERS'

from datasets import load_dataset, Audio
import torch
import glob, librosa
from tqdm import tqdm
import numpy as np

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
    start = time.time()
    outputs = llm.generate(prompts,sampling_params)
    duration = time.time() - start
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
            audio_files = glob.glob(os.path.join(wav_path, f'*{row['aid']}*.wav'))
            assert len(audio_files) == 1
            audio_file = audio_files[0]
            audio, sample_rate = librosa.load(audio_file,sr=None)
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

if __name__ == "__main__":
    # https://huggingface.co/docs/datasets/access
    #dataset = load_demo_datasets()
    # 
    #dataset = load_1000_datasets()

    # HF wer_score: 1.74 %
    # Accuracy is consistent in both AMD & NV
    # vLLM wer_score batch-size1  ,kv-cache-fp16: 1.88 %
    # vLLM wer_score batch-size1  ,kv-cache-fp8 : 1.88 %
    # vLLM wer_score batch-size256,kv-cache-fp8 : 1.88 %
    dataset = load_librispeech_asr_test_datasets()
    dataset = dataset.filter(lambda example: example["audio"]["array"].shape[0] < 30*16000)

    print(dataset)

    #predictions = speech_recognition_hf(dataset)
    predictions = speech_recognition_vllm(dataset, use_fp8_kv_cache=True, batch_size=256)
    # https://pypi.org/project/whisper-normalizer/
    from whisper_normalizer.english import EnglishTextNormalizer
    english_normalizer = EnglishTextNormalizer()
    predictions = [english_normalizer(x) for x in predictions]
    references = [english_normalizer(dataset[i]["text"]) for i in range(dataset.num_rows)]

    import evaluate
    wer = evaluate.load("wer")
    wer_score = wer.compute(predictions=predictions, references=references)
    print(f"wer_score: {wer_score * 100:.2f} %")
