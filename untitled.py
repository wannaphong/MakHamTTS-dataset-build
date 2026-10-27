import json
import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastthaig2p import G2P, ipa_to_kokoro

g2p = G2P()
cfg = json.load(open("paradee/models/paradee_thai/config.json", encoding="utf-8"))
sess = ort.InferenceSession("paradee/models/paradee_thai/paradee_thai_int8.onnx", providers=["CPUExecutionProvider"])

text = "สวัสดีค่ะ ยินดีต้อนรับสู่บริการของเรา"
phonemes = ipa_to_kokoro(g2p.convert(text))
ids = [0] + [cfg["vocab"][c] for c in phonemes if c in cfg["vocab"]] + [0]

audio = sess.run(None, {
    "input_ids": np.array([ids], dtype=np.int64),
    "speed": np.array([1.0], dtype=np.float32),
})[0][0]

sf.write("output.wav", audio, 24000)