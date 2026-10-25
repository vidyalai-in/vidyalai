from tokenizer import BPETokenizer

tok = BPETokenizer()
tok.train(["the cat sat on the mat. the cat ate the rat."], vocab_size=256 + 6 + 8)
ids = tok.encode("the cat ate the hat.<|endoftext|>")
print(ids)                           # [258, 261, 32, 256, 101, 259, 32, 104, 256, 46, 262]
print([tok.decode([i]) for i in ids])
# ['the', ' cat', ' ', 'at', 'e', ' the', ' ', 'h', 'at', '.', '<|endoftext|>']
