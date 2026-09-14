import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer, AutoConfig
from huggingface_hub import hf_hub_download

repo_id = "videosdk-live/Namo-Turn-Detector-v1-Japanese"
model_path = hf_hub_download(repo_id=repo_id, filename="model_quant.onnx")
tok = AutoTokenizer.from_pretrained(repo_id)
cfg = AutoConfig.from_pretrained(repo_id)
print("Config id2label:", getattr(cfg, "id2label", None))
print("Config label2id:", getattr(cfg, "label2id", None))

sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
sentences = [
    "1382年に聖パウロ修道会のために建てられた僧院です。",
    "1913年マニラで第1回東洋オリンピックが開会だから",
]
for s in sentences:
    inp = tok(s, return_tensors="np", truncation=True, max_length=512)
    fd = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
    logits = sess.run(None, fd)[0][0]
    # Softmax
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    print(f"Text: '{s}' -> logits={logits.tolist()} probs={probs.tolist()}")

# Also check Multilingual config
print("\n--- Multilingual ---")
multi_repo = "videosdk-live/Namo-Turn-Detector-v1-Multilingual"
multi_cfg = AutoConfig.from_pretrained(multi_repo)
print("Multilingual id2label:", getattr(multi_cfg, "id2label", None))
print("Multilingual label2id:", getattr(multi_cfg, "label2id", None))
multi_model_path = hf_hub_download(repo_id=multi_repo, filename="model_quant.onnx")
multi_tok = AutoTokenizer.from_pretrained(multi_repo)
multi_sess = ort.InferenceSession(multi_model_path, providers=["CPUExecutionProvider"])
for s in sentences:
    inp = multi_tok(s, return_tensors="np", truncation=True, max_length=8192)
    fd = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
    logits = multi_sess.run(None, fd)[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    print(f"Text: '{s}' -> logits={logits.tolist()} probs={probs.tolist()}")
