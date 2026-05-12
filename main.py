from k1lib.imports import *
from paddleocr import PaddleOCR
from flask import request
import tempfile

def text_genFast(text:str) -> str:
    return requests.post(f"http://localhost:11434/api/chat", json={ "model": "qwen2.5:3b", "messages": [{"role": "user", "content": text}], "think": False, "stream": False })\
        .json()["message"]["content"].strip()

app = web.Flask(__name__)
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_recognition_batch_size=1, textline_orientation_batch_size=1, text_det_limit_side_len=2000, text_det_limit_type="max", enable_hpi=False, use_tensorrt=False, precision="fp32")

def doOcr(im):
    fn = tempfile.mkstemp(suffix=".jpg")[1]; im | toBytes() | file(fn); a = time.time(); d = ocr.predict(input=fn)[0]; b = time.time() #  | filt("x>0.5", 2)
    t = [d["rec_texts"], d["rec_boxes"], d["rec_scores"]] | T() | ~apply(lambda x1,y1,x2,y2: [x1,y1,x2,y2,(x1+x2)/2, (y1+y2)/2], 1) | apply(round, 2, ndigits=2) | ~apply(lambda t,bb,s: {"text": t, "bbox": bb[:4], "center": bb[4:], "score": s}) | deref()
    w, h = im | shape(); os.remove(fn); return {"width": w, "height": h, "data": t, "execDuration": b-a, "ver": 2}

@app.route("/receive", methods=["POST"])
def receive(): return json.dumps(doOcr(request.data | toImg()))

def hasNonEnglish(text: str) -> bool:
    for ch in text: # ASCII printable range (space to ~)
        if not (32 <= ord(ch) <= 126): return True
    return False

def postprocess(x): return x.strip("'").strip('"').strip().strip("`").split("\n") | filt("x") | item()

@app.route("/translate", methods=["POST"])
def translate():
    im = request.data | toImg(); ocr = doOcr(im)["data"]
    prompt1 = "Is this english? ANSWER TRUE/FALSE ONLY\n\n"
    prompt2 = "Translate the text below to english, be as succinct as possible. DO NOT EXPLAIN, DO NOT ADD FILLER TEXT, DO NOT ADD NOTES, EITHER TRANSLATE OR DO NOTHING\n\n"
    for i,o in enumerate(ocr):
        if hasNonEnglish(o["text"]):
            tf = text_genFast(prompt1 + o["text"] + "\n\n") | aS(postprocess)
            if tf.lower().startswith("f"): o["text"] = text_genFast(prompt2 + o["text"] + "\n\n") | aS(postprocess)
    return k1.drawBoxesOnImage(im, transOldDrawBoxesFormat(ocr), lineSpacing=0) | toBytes()

def transOldDrawBoxesFormat(ocr):
    res = []
    for o in ocr: bb = o["bbox"]; res.append([*o["center"], bb[2] - bb[0], bb[3] - bb[1], o["text"]])
    return res

@app.route("/")
def index(): return "ok"

app.run(host="0.0.0.0", port=5005)

    
