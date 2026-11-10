"""CLIP istemlerini iyileştir: alternatif İngilizce ifadelerle çok-istemli (ensemble) skor.

Sorun: tek istem ("a book") dik tutulan kitapta zayıf; CLIP onu "vazo" sanıyor.
Çözüm: her sınıf için birkaç doğal ifade ("a book cover", "a hardcover book",
"a closed book") kullanıp skorları birleştirmek. Bu, eğitim gerektirmez ve
CLIP'in sıfır-atış performansını belirgin biçimde artırır.

Ölçüm (bench_images, 6 görsel):
    tek istem     : 4/6
    çok-istem     : bu betikle ölçülür
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import open_clip
import torch
from PIL import Image

#: Türkçe sınıf -> birden fazla İngilizce istem (ensemble).
ENSEMBLE: dict[str, list[str]] = {
    "su ısıtıcısı": ["an electric kettle", "a kettle for boiling water", "a water boiler appliance"],
    "kitap": [
        "a book",
        "a book cover standing upright",
        "a hardcover book",
        "a closed book",
        "a book standing on a shelf",
    ],
    "fincan": ["a coffee mug", "a cup of coffee", "a ceramic mug", "a tea cup"],
    "dizüstü bilgisayar": [
        "a laptop computer",
        "an open laptop on a desk",
        "a notebook computer",
        "a laptop screen and keyboard",
    ],
    "şarap kadehi": ["a wine glass", "a glass of wine", "a stemmed wine glass"],
    "şişe": ["a bottle", "a plastic bottle", "a glass bottle"],
    "kase": ["a bowl", "a ceramic bowl", "a bowl of food"],
    "tencere": ["a cooking pot", "a saucepan", "a pot on a stove"],
    "sürahi": ["a water pitcher", "a jug of water", "a carafe"],
    "termos": ["a thermos flask", "a travel mug", "an insulated bottle"],
    "vazo": ["a vase", "a flower vase", "a decorative vase"],
    "saksı bitkisi": ["a potted plant", "a plant in a pot", "a houseplant"],
    "kalem": ["a pen", "a ballpoint pen", "a writing pen"],
    "telefon": ["a mobile phone", "a smartphone", "a cell phone"],
    "klavye": ["a computer keyboard", "a keyboard on a desk"],
    "monitör": ["a computer monitor", "a computer screen", "a display monitor"],
}

EXPECTED = {
    "sup_kettle": "su ısıtıcısı",
    "kitap_dik": "kitap",
    "kitap_yatik": "kitap",
    "sarap_kadehi": "şarap kadehi",
    "fincan": "fincan",
    "laptop": "dizüstü bilgisayar",
}


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_images")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    labels = list(ENSEMBLE)
    # Sınıf başına ortalama: her istem ayrı vektör, sonra normalize ortalama.
    prompts: list[str] = []
    owner: list[int] = []
    for index, label in enumerate(labels):
        for prompt in ENSEMBLE[label]:
            prompts.append(prompt)
            owner.append(index)

    with torch.no_grad():
        features = model.encode_text(tokenizer(prompts))
        features /= features.norm(dim=-1, keepdim=True)
        # Sınıf başına birleşik vektör (ortalama + normalize).
        class_features = torch.zeros(len(labels), features.shape[1])
        for index in range(len(labels)):
            mask = torch.tensor([o == index for o in owner])
            merged = features[mask].mean(dim=0)
            class_features[index] = merged / merged.norm()

    hits = 0
    total = 0
    for name, expected in EXPECTED.items():
        frame = cv2.imread(str(folder / f"{name}.jpg"))
        if frame is None:
            continue
        image = preprocess(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            probs = (100.0 * image_features @ class_features.T).softmax(dim=-1)[0]

        top = torch.topk(probs, 3)
        ranked = [(labels[i], float(probs[i])) for i in top.indices.tolist()]
        best_label, best_score = ranked[0]
        ok = best_label == expected
        hits += 1 if ok else 0
        total += 1
        alts = ", ".join(f"{label}:{score:.2f}" for label, score in ranked[1:])
        print(
            f"  {'OK ' if ok else '  X'} {name:14s} -> {best_label:20s} "
            f"({best_score:.2f})  alt: {alts}"
        )

    print(f"\nçok-istem doğru: {hits}/{total}")


if __name__ == "__main__":
    main()
