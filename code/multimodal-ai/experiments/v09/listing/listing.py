import jiwer, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

proc = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small", dtype=torch.float16).to("cuda")

audio, sr = sf.read("speech.wav", dtype="float32")      # mono, 16 kHz
x = proc(audio, sampling_rate=sr, return_tensors="pt").input_features
print(x.shape)                                         # 80 mel bins x 3,000 frames
ids = model.generate(x.to("cuda", torch.float16), language="en", task="transcribe")
text = proc.batch_decode(ids, skip_special_tokens=True)[0]
print(text)

ref = "Whisper listens to thirty seconds at a time and writes down what it heard."
basic = BasicTextNormalizer()                          # lowercase, drop punctuation
print(jiwer.wer(basic(ref), basic(text)))
english = proc.tokenizer.normalize                     # Whisper's English normalizer
print(jiwer.wer(english(ref), english(text)))
