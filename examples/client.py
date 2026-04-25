"""
Example client — run against a live server:
    python examples/client.py --host http://localhost:8000
"""
import argparse
import json
import time
import urllib.request


def post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.host.rstrip("/")

    with urllib.request.urlopen(f"{base}/health") as r:
        print("Health:", json.loads(r.read()))

    print("\n── Single classify ──")
    for text in ["This movie was amazing!", "Worst experience ever."]:
        t0 = time.perf_counter()
        result = post(f"{base}/v1/classify", {"text": text})
        dt = (time.perf_counter() - t0) * 1000
        print(f"  text='{text}' -> {result['label']} (conf={result['confidence']:.3f}, "
              f"tokens={result['input_tokens']}, {dt:.1f}ms)")

    print("\n── Batch classify ──")
    texts = [
        "I loved every minute of it.",
        "Painfully boring and predictable.",
        "An absolute masterpiece.",
        "Not the worst, but not great.",
    ]
    t0 = time.perf_counter()
    batch = post(f"{base}/v1/classify/batch", {"texts": texts})
    dt = (time.perf_counter() - t0) * 1000
    for text, item in zip(texts, batch["outputs"]):
        print(f"  '{text[:40]:<40}' -> {item['label']} (conf={item['confidence']:.3f})")
    print(f"Batch latency: {dt:.1f} ms")


if __name__ == "__main__":
    main()
