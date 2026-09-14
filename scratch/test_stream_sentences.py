import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

def test_stream_sentences():
    # Sentences and incomplete prefixes from 00_ingress_stream
    test_cases = [
        # (text, is_complete_truth)
        ("このホテル初めてですよね。", True),
        ("このホテル初めて", False),
        ("温泉がいいらしいよ。", True),
        ("温泉がいい", False),
        ("相変わらず何も調べてないんだ。", True),
        ("相変わらず何も", False),
        ("だろ。", True),
        ("部屋どうでした？", True),
        ("部屋どう", False),
        ("私はふすまがあれば別に大丈夫ですよ。", True),
        ("私はふすまがあれば", False),
        ("彼はずっとそういう人だった。", True),
        ("彼はずっと", False),
    ]

    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    models = [
        ("Japanese (Specialized)", "videosdk-live/Namo-Turn-Detector-v1-Japanese", 512),
        ("Multilingual (Unified)", "videosdk-live/Namo-Turn-Detector-v1-Multilingual", 8192),
    ]

    for model_title, repo_id, max_len in models:
        print(f"\n==========================================")
        print(f"Model: {model_title} ({repo_id})")
        print(f"==========================================")
        mp = hf_hub_download(repo_id=repo_id, filename="model_quant.onnx")
        tok = AutoTokenizer.from_pretrained(repo_id)
        sess = ort.InferenceSession(mp, providers=["CPUExecutionProvider"])

        correct = 0
        for text, truth in test_cases:
            inp = tok(text, return_tensors="np", truncation=True, max_length=max_len)
            fd = {"input_ids": inp["input_ids"].astype(np.int64), "attention_mask": inp["attention_mask"].astype(np.int64)}
            logits = sess.run(None, fd)[0][0]
            probs = softmax(logits)
            pred = int(np.argmax(probs))
            conf = float(probs[1]) # probability of EOU (class 1)
            is_eou = (pred == 1)
            is_match = (is_eou == truth)
            if is_match:
                correct += 1
            mark = "OK" if is_match else "FAIL"
            print(f"[{mark}] Pred EOU={is_eou} (p_eou={conf:.3f}) | Truth EOU={truth:<5} | '{text}'")
        print(f"Accuracy: {correct}/{len(test_cases)} ({correct/len(test_cases)*100:.1f}%)")

if __name__ == "__main__":
    test_stream_sentences()
