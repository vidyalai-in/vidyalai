import sys, time, torch
sys.path.insert(0, "/home/amit/.cache/vidyalai-scratch/mm/lab"); import vlm
m = vlm.VLM(sys.argv[1])
it = vlm.make_items(5)[0]
for size in (224, 896):
    img = vlm.render(it, size); q = it.question
    for rep in range(3):
        t0=time.time(); x = m.inputs(img, q); torch.cuda.synchronize(); t1=time.time()
        with torch.inference_mode(): out = m.model.generate(**x, max_new_tokens=8, do_sample=False)
        torch.cuda.synchronize(); t2=time.time()
        with torch.inference_mode(): m.model(**x, logits_to_keep=1)
        torch.cuda.synchronize(); t3=time.time()
        print(size, 'proc %.3f gen %.3f prefill %.3f newtok %d' % (t1-t0, t2-t1, t3-t2, out.shape[1]-x['input_ids'].shape[1]))
