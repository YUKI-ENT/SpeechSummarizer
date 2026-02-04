from faster_whisper import WhisperModel

# ここだけ変える
MODEL = "kotoba-tech/kotoba-whisper-v1.0-faster"
DEVICE = "cuda"
COMPUTE = "float16"

print("Downloading / loading model...", MODEL)
m = WhisperModel(MODEL, device=DEVICE, compute_type=COMPUTE)
print("OK")
