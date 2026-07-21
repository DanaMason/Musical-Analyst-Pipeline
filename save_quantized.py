import pipeline as P
model, tok = P.load_phi4()
out = "/home/ren-admin/sdso/phi4-4bit"
model.save_pretrained(out)
tok.save_pretrained(out)