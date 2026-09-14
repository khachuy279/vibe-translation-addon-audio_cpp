import sys
import time
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def test_namo_model(repo_id: str, test_sentences: list[str], max_length: int = 512):
    print(f"\n==========================================")
    print(f"Testing Namo model from: {repo_id}")
    print(f"==========================================")
    t0 = time.perf_counter()
    model_path = hf_hub_download(repo_id=repo_id, filename="model_quant.onnx")
    download_time = time.perf_counter() - t0
    print(f"Downloaded / cached model path: {model_path} ({download_time:.2f}s)")
    
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    tok_time = time.perf_counter() - t0
    print(f"Loaded tokenizer in {tok_time:.2f}s")
    
    t0 = time.perf_counter()
    # Configure ONNX Runtime session
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    sess_time = time.perf_counter() - t0
    print(f"Loaded ONNX session in {sess_time:.2f}s")

    for input_meta in session.get_inputs():
        print(f"  Input: name={input_meta.name}, shape={input_meta.shape}, type={input_meta.type}")
    for output_meta in session.get_outputs():
        print(f"  Output: name={output_meta.name}, shape={output_meta.shape}, type={output_meta.type}")

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    latencies = []
    for s in test_sentences:
        t_start = time.perf_counter()
        inputs = tokenizer(
            s,
            truncation=True,
            max_length=max_length,
            return_tensors="np"
        )
        feed_dict = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64)
        }
        outputs = session.run(None, feed_dict)
        logits = outputs[0]
        probs = softmax(logits[0])
        pred_label = int(np.argmax(probs))
        conf = float(np.max(probs))
        infer_ms = (time.perf_counter() - t_start) * 1000.0
        latencies.append(infer_ms)
        status = "EOU (End of Turn)" if pred_label == 1 else "Incomplete (Continuation)"
        print(f"Text: '{s}' -> {status} (conf: {conf:.4f}, infer: {infer_ms:.2f}ms)")

    print(f"Average latency: {np.mean(latencies):.2f}ms | Min: {np.min(latencies):.2f}ms | Max: {np.max(latencies):.2f}ms")

if __name__ == "__main__":
    ja_sentences = [
        "1382年に聖パウロ修道会のために建てられた僧院です。",
        "1913年マニラで第1回東洋オリンピックが開会だから",
        "こんにちは、本日の天気は",
        "ありがとうございます。"
    ]
    # Test Japanese specialized model
    test_namo_model("videosdk-live/Namo-Turn-Detector-v1-Japanese", ja_sentences, max_length=512)
    
    # Test Multilingual model
    multi_sentences = [
        "Hello, how can I help you today?",
        "I was thinking that maybe we could",
        "1382年に聖パウロ修道会のために建てられた僧院です。",
        "1913年マニラで第1回東洋オリンピックが開会だから",
    ]
    test_namo_model("videosdk-live/Namo-Turn-Detector-v1-Multilingual", multi_sentences, max_length=8192)
