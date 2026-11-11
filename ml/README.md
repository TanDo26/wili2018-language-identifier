# ml/ — algorithm & evaluation (owner: Anh Phan)

## For the API (Ngọc Bảo)
```python
from predict import predict
predict(["Con sông chảy lặng lẽ qua phố cổ."], top_k=3, model_dir="models/v1")
```
Returns one dict per text:
```json
{"label": "vie", "iso": "vie", "confidence": 0.93,
 "candidates": [{"label": "vie", "iso": "vie", "confidence": 0.93}, "..."],
 "model_version": "v1"}
```
- `confidence` of the NB model is a probability. For an SVM it is softmax of decision scores (ranking only, not calibrated).
- Short or empty text: validate in the API (< 20 chars is rejected by your rules); the model will still return something.
- Load once and reuse: `load(model_dir)` is cached. To switch versions without redeploy, call `predict(..., model_dir="models/vN")`.

## Model folder layout: `models/vN/`
`vectorizer.joblib`, `model.joblib`, `config.json` (version, model, hyperparameter, labels, iso_codes, trained_on, trained_at), `metrics.json` (val/test accuracy, macro-F1, confused pairs, latency).

## Try it now
`python smoke_test.py` creates `models/stub/`, a tiny STUB model (12 sentences, not WiLI-2018) so the API can integrate before the real `models/v1/` arrives. Its metrics are meaningless. `samples/metrics.sample.json` has FAKE numbers for the console and API.

## Run (ML side)
```
pip install -r requirements.txt
python train.py --model majority --out experiments/majority
python train.py --model nb  --tune --version v1
python train.py --model svm --tune --out experiments/svm
python evaluate.py models/v1 --split val
python evaluate.py models/v1 --split test      # once only
```
Splits: if `data/split/{train,val,test}.parquet` (id, text, label) exist they are used; otherwise a TEMPORARY 90/10 split is made.
