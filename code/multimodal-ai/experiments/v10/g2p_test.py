from kokoro import KPipeline
p = KPipeline(lang_code='a', model=False)
for t in ["I read the book yesterday, and I will read it again.", "The bill was $1,234.56 on Jan. 5, 2024.", "Dr. Smith lives on St. James St.", "The quokka met a zyzzyva near Llanfairpwll.", "Please record the record.", "It's 3:45 pm and 72°F."]:
    for gs, ps, _ in p(t):
        print(repr(gs), '->', ps)
