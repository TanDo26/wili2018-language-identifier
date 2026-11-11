"""Builds a tiny STUB model (models/stub/) from a few hand-written sentences, to test the
pipeline offline and let the API start integrating. NOT the real v1; numbers are meaningless."""
from train import run
from evaluate import evaluate
from predict import predict

S = {
 "eng": ["The river flows quietly through the old town.", "She reads a book every evening before bed.",
         "We are going to the market to buy some fresh fruit.", "The weather today is warm and the sky is clear.",
         "He explained the problem to the whole class again."],
 "vie": ["Con sông chảy lặng lẽ qua phố cổ.", "Cô ấy đọc một cuốn sách mỗi tối trước khi đi ngủ.",
         "Chúng tôi đi chợ để mua một ít trái cây tươi.", "Hôm nay trời ấm và bầu trời rất trong xanh.",
         "Anh ấy giải thích lại vấn đề cho cả lớp nghe."],
 "fra": ["La rivière traverse doucement la vieille ville.", "Elle lit un livre chaque soir avant de dormir.",
         "Nous allons au marché pour acheter des fruits frais.", "Il fait chaud aujourd'hui et le ciel est dégagé.",
         "Il a expliqué de nouveau le problème à toute la classe."],
 "deu": ["Der Fluss fließt ruhig durch die Altstadt.", "Sie liest jeden Abend ein Buch vor dem Schlafen.",
         "Wir gehen auf den Markt, um frisches Obst zu kaufen.", "Das Wetter ist heute warm und der Himmel klar.",
         "Er erklärte der ganzen Klasse das Problem noch einmal."]}

def split(sl):
    X = [s for l, ss in S.items() for s in ss[sl]]; y = [l for l, ss in S.items() for _ in ss[sl]]
    return X, y

data = {"train": split(slice(0, 3)), "val": split(slice(3, 4)), "test": split(slice(4, 5)),
        "source": "STUB: 12 hand-written sentences (not WiLI-2018)"}
out = run(data, "nb", "stub", "models/stub")
print(predict(["Hôm nay tôi đi học."], top_k=2, model_dir=out))
print({k: v for k, v in evaluate(out, "val", data).items() if k in ("accuracy", "macro_f1", "latency_ms_median")})
