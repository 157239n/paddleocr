from k1lib.imports import *
from paddleocr import PaddleOCR
from flask import request
import tempfile

def text_genFast(text:str) -> str:
    return requests.post(f"http://localhost:11434/api/chat", json={ "model": "qwen2.5:3b", "messages": [{"role": "user", "content": text}], "think": False, "stream": False })\
        .json()["message"]["content"].strip()

app = web.Flask(__name__)
ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, text_recognition_batch_size=1, textline_orientation_batch_size=1, text_det_limit_side_len=1280, text_det_limit_type="max", enable_hpi=False, use_tensorrt=False, precision="fp16")

def doOcr(im):
    fn = tempfile.mkstemp(suffix=".jpg")[1]; im | toBytes() | file(fn); a = time.time(); d = ocr.predict(input=fn)[0]; b = time.time()
    t = [d["rec_texts"], d["rec_boxes"], d["rec_scores"]] | T() | ~apply(lambda x1,y1,x2,y2: [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], 1) | filt("x>0.9", 2) | apply(round, 2, ndigits=2) | insert(["text", "cx,cy,w,h", "score"]) | deref()
    w, h = im | shape(); os.remove(fn); return {"width": w, "height": h, "data": t, "execDuration": b-a}

@app.route("/receive", methods=["POST"])
def receive(): return json.dumps(doOcr(request.data | toImg()))

def hasNonEnglish(text: str) -> bool:
    for ch in text: # ASCII printable range (space to ~)
        if not (32 <= ord(ch) <= 126): return True
    return False

def postprocess(x): return x.strip("'").strip('"').strip().strip("`").split("\n") | filt("x") | item()

@app.route("/translate", methods=["POST"])
def translate():
    im = request.data | toImg(); ocr = doOcr(im)["data"] | cut(0, 1) | ~apply(lambda x,y: [*y,x]) | ~head(1) | aS(list)
    prompt1 = "Is this english? ANSWER TRUE/FALSE ONLY\n\n"
    prompt2 = "Translate the text below to english, be as succinct as possible. DO NOT EXPLAIN, DO NOT ADD FILLER TEXT, DO NOT ADD NOTES, EITHER TRANSLATE OR DO NOTHING\n\n"
    for i,o in enumerate(ocr):
        if hasNonEnglish(o[4]):
            tf = text_genFast(prompt1 + o[4] + "\n\n") | aS(postprocess)
            if tf.lower().startswith("f"): o[4] = text_genFast(prompt2 + o[4] + "\n\n") | aS(postprocess)
    return k1.drawBoxesOnImage(im, ocr, lineSpacing=0) | toBytes()

@app.route("/")
def index(): return "ok"

app.run(host="0.0.0.0", port=5005)

    
