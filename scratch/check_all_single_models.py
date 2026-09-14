import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

models = {
    "en": ("videosdk-live/Namo-Turn-Detector-v1-English", ["They're often made with oil or sugar.", "I think the next logical step is to"]),
    "zh": ("videosdk-live/Namo-Turn-Detector-v1-Chinese", ["今天天气真好，我们出去散步吧。", "如果明天不下雨的话我们就"]),
    "ko": ("videosdk-live/Namo-Turn-Detector-v1-Korean", ["오늘 날씨가 정말 좋네요.", "내일 시간이 되면 같이"]),
    "ru": ("videosdk-live/Namo-Turn-Detector-v1-Russian", ["Сегодня отличная погода для прогулки.", "Если завтра не будет дождя то мы"]),
}

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

for lang, (repo_id, test_sentences) in models.items():
    print(f"\n--- Checking {lang}: {repo_id} ---", flush=True)
    try:
        t0 = time.perf_counter()
        mp = hf_hub_download(repo_id=repo_id, filename="model_quant.onnx")
        tok = AutoTokenizer.from_pretrained(repo_id)
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])
        load_time = time.perf_counter() - t0
        print(f"Loaded in {load_time:.2f}s", flush=True)
        for s in test_sentences:
            inp = tok(s, return_tensors="np", truncation=True, max_length=512)
            fd = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
            logits = sess.run(None, fd)[0][0]
            probs = softmax(logits)
            pred = int(np.argmax(probs))
            conf = float(np.max(probs))
            status = "EOU (End of Turn)" if pred == 1 else "Incomplete (Continuation)"
            print(f"  '{s}' -> {status} (label={pred}, conf={conf:.3f}, probs={probs.tolist()})", flush=True)
    except Exception as e:
        print(f"  ERROR for {lang}: {e}", flush=True)
