from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizer import PATTERN, SPECIAL_TOKENS
from train_tokenizer import read_docs       # streams the 406 MB sample, one document at a time

tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
])
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=32_768,
    special_tokens=SPECIAL_TOKENS,                         # ids 0..7
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # all 256 bytes, ids 8..263
)
tok.train_from_iterator(read_docs("train"), trainer)       # 32,504 merges, ids 264..32767
tok.save("tokenizer-32k.json")
