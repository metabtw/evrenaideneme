"""İstem stratejisi karşılaştırması: tek istem vs. vazosuz katalog vs. ensemble.

Amaç: fincan/kitap karışmasının kaynağını izole etmek. "vazo" istemi hem
kadehi hem fincanı çekiyor; katalogda onu çıkarınca ne oluyor?
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import open_clip
import torch
from PIL import Image

EXPECTED = {
    "sup_kettle": "su ısıtıcısı",
    "kitap_dik": "kitap",
    "kitap_yatik": "kitap",
    "sarap_kadehi": "şarap kadehi",
    "fincan": "fincan",
    "laptop": "dizüstü bilgisayar",
}

SCENARIOS: dict[str, dict[str, str]] = {
    "A: geniş katalog (16)": {
        "su ısıtıcısı": "an electric kettle",
        "kitap": "a book",
        "fincan": "a coffee mug",
        "dizüstü bilgisayar": "a laptop computer",
        "şarap kadehi": "a wine glass",
        "şişe": "a bottle",
        "kase": "a bowl",
        "tencere": "a cooking pot",
        "sürahi": "a water pitcher",
        "termos": "a thermos flask",
        "vazo": "a vase",
        "saksı bitkisi": "a potted plant",
        "kalem": "a pen",
        "telefon": "a mobile phone",
        "klavye": "a computer keyboard",
        "monitör": "a computer monitor",
    },
    "B: vazosuz (15)": {
        "su ısıtıcısı": "an electric kettle",
        "kitap": "a book",
        "fincan": "a coffee mug",
        "dizüstü bilgisayar": "a laptop computer",
        "şarap kadehi": "a wine glass",
        "şişe": "a bottle",
        "kase": "a bowl",
        "tencere": "a cooking pot",
        "sürahi": "a water pitcher",
        "termos": "a thermos flask",
        "saksı bitkisi": "a potted plant",
        "kalem": "a pen",
        "telefon": "a mobile phone",
        "klavye": "a computer keyboard",
        "monitör": "a computer monitor",
    },
    "C: sadece 6 sınıf": {
        "su ısıtıcısı": "an electric kettle",
        "kitap": "a book",
        "fincan": "a coffee mug",
        "dizüstü bilgisayar": "a laptop computer",
        "şarap kadehi": "a wine glass",
        "şişe": "a bottle",
    },
    "D: kitap+vazo ayrımı güçlendirilmiş": {
        "su ısıtıcısı": "an electric kettle",
        "kitap": "a book with a cover",
        "fincan": "a coffee mug",
        "dizüstü bilgisayar": "a laptop computer",
        "şarap kadehi": "a wine glass",
        "şişe": "a bottle",
        "vazo": "an empty flower vase",
        "kase": "a bowl",
        "sürahi": "a water pitcher",
        "termos": "a thermos flask",
        "kalem": "a pen",
        "telefon": "a mobile phone",
        "klavye": "a computer keyboard",
        "monitör": "a computer monitor",
        "saksı bitkisi": "a potted plant",
        "tencere": "a cooking pot",
    },
}


def main() -> None:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_images")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    images = {}
    for name in EXPECTED:
        frame = cv2.imread(str(folder / f"{name}.jpg"))
        if frame is not None:
            images[name] = preprocess(Image.fromarray(frame[:, :, ::-1])).unsqueeze(0)

    with torch.no_grad():
        image_features = {}
        for name, tensor in images.items():
            features = model.encode_image(tensor)
            image_features[name] = features / features.norm(dim=-1, keepdim=True)

    for scenario, prompts in SCENARIOS.items():
        labels = list(prompts)
        with torch.no_grad():
            text = model.encode_text(tokenizer(list(prompts.values())))
            text /= text.norm(dim=-1, keepdim=True)

        hits = 0
        wrong: list[str] = []
        for name, expected in EXPECTED.items():
            if name not in image_features:
                continue
            with torch.no_grad():
                probs = (100.0 * image_features[name] @ text.T).softmax(dim=-1)[0]
            best = labels[int(torch.argmax(probs))]
            if best == expected:
                hits += 1
            else:
                wrong.append(f"{name}->{best}({float(probs.max()):.2f})")

        print(f"{scenario:38s} {hits}/{len(images)}")
        if wrong:
            print(f"    hatalar: {', '.join(wrong)}")


if __name__ == "__main__":
    main()
