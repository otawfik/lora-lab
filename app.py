"""Flask web UI: type a prompt, get base vs pirate generations side by side.

Usage:
    python app.py
Then open http://127.0.0.1:5000 in a browser. Run train.py first so the
weights files exist.
"""

import os

from flask import Flask, request, render_template_string

from loralab.adapter import LoRAAdapter
from loralab.model import TinyLM
from loralab.tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")

app = Flask(__name__)

TOKENIZER = None
BASE = None
ADAPTER = None
LOAD_ERROR = None

try:
    TOKENIZER = Tokenizer.load(os.path.join(WEIGHTS, "vocab.json"))
    BASE = TinyLM.load(os.path.join(WEIGHTS, "base.npz"))
    ADAPTER = LoRAAdapter.load(os.path.join(WEIGHTS, "lora_adapter.npz"))
except Exception as exc:  # missing weights: show a helpful message instead
    LOAD_ERROR = str(exc)

PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LoRA Lab: base vs pirate</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto;
           padding: 0 1rem; background: #0f172a; color: #e2e8f0; }
    h1 { font-size: 1.6rem; }
    .sub { color: #94a3b8; margin-bottom: 1.5rem; }
    form { background: #1e293b; padding: 1rem; border-radius: 8px; }
    input[type=text] { width: 100%; padding: .6rem; font-size: 1rem;
                       border-radius: 6px; border: 1px solid #475569;
                       background: #0f172a; color: #e2e8f0; }
    .row { display: flex; gap: 1rem; margin-top: .8rem; align-items: center; flex-wrap: wrap; }
    label { color: #94a3b8; font-size: .9rem; }
    button { padding: .6rem 1.2rem; font-size: 1rem; border-radius: 6px; border: 0;
             background: #38bdf8; color: #0f172a; cursor: pointer; font-weight: 600; }
    .panels { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
    .panel { background: #1e293b; padding: 1rem; border-radius: 8px; }
    .panel h2 { margin-top: 0; font-size: 1.1rem; }
    .base h2 { color: #7dd3fc; } .pirate h2 { color: #fbbf24; }
    .out { line-height: 1.6; white-space: pre-wrap; }
    .err { background: #7f1d1d; padding: 1rem; border-radius: 8px; }
    @media (max-width: 640px) { .panels { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>LoRA Lab</h1>
  <div class="sub">One tiny model, two personalities. The base model learned plain
  English, then a LoRA-style adapter (rank 8, under 10% of base params) taught it to
  talk like a pirate. Same prompt, both answers.</div>
  {% if load_error %}
    <div class="err">Weights not found ({{ load_error }}). Run
    <code>python train.py</code> first, then reload.</div>
  {% endif %}
  <form method="post">
    <input type="text" name="prompt" value="{{ prompt }}" placeholder="Type a prompt, e.g. the treasure is">
    <div class="row">
      <label>Temperature <input type="range" name="temperature" min="0.3" max="1.5"
             step="0.1" value="{{ temperature }}"></label>
      <label>Words <input type="number" name="n_tokens" min="10" max="80"
             value="{{ n_tokens }}" style="width:4rem"></label>
      <button type="submit">Generate</button>
    </div>
  </form>
  {% if base_out %}
  <div class="panels">
    <div class="panel base"><h2>Base model</h2><div class="out">{{ base_out }}</div></div>
    <div class="panel pirate"><h2>Pirate adapter</h2><div class="out">{{ pirate_out }}</div></div>
  </div>
  {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    prompt = "the treasure is"
    temperature = 0.9
    n_tokens = 40
    base_out = pirate_out = None
    if request.method == "POST":
        prompt = request.form.get("prompt", prompt).strip() or prompt
        temperature = float(request.form.get("temperature", temperature))
        n_tokens = int(request.form.get("n_tokens", n_tokens))
        if TOKENIZER is not None:
            ids = TOKENIZER.encode(prompt)
            if ids:
                gen_kw = dict(n_tokens=n_tokens, temperature=temperature,
                              seed=11, bos_id=TOKENIZER.bos_id,
                              suppress_ids=(TOKENIZER.unk_id,),
                              stop_on=(TOKENIZER.eos_id,))
                base_out = TOKENIZER.decode(ids + BASE.generate(ids, **gen_kw))
                pirate_out = TOKENIZER.decode(
                    ids + ADAPTER.generate(ids, **gen_kw))
            else:
                base_out = pirate_out = "(prompt had no known words, try again)"
    return render_template_string(
        PAGE, prompt=prompt, temperature=temperature, n_tokens=n_tokens,
        base_out=base_out, pirate_out=pirate_out, load_error=LOAD_ERROR)


if __name__ == "__main__":
    app.run(debug=False)
