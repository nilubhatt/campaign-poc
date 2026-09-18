"""
Convert the CLIP checkpoint to fp16 and prove the conversion is safe (plan item 1.2b).

The fp32 checkpoint is 605MB, which lands a ~780MB Setup.exe in front of a customer behind
a corporate proxy doing AV inspection. fp16 halves it. The review suggested this with the
condition "no meaningful accuracy cost for similarity ranking" — a condition, not a given,
so this script measures it rather than asserting it.

    python scripts/convert_fp16.py <fp32.safetensors> <out.safetensors> [photo_dir]

Pass a directory of real photographs to measure the case that actually matters; without
one it looks for the photo sets macOS and most Linux desktops ship, and falls back to
synthetic shapes (which measure the easy case and flatter the result).

Loading fp16 weights into the fp32 model is fine on CPU: torch's load_state_dict copies
with dtype conversion, so the model still computes in fp32 — only the stored file is
halved. That is verified below, not assumed.
"""
from __future__ import annotations

import sys
from pathlib import Path


def convert(src: Path, dest: Path) -> Path:
    """Cast every floating-point tensor to fp16. Integer buffers are left alone."""
    from safetensors.torch import load_file, save_file

    state = load_file(str(src))
    out = {
        key: (tensor.half() if tensor.is_floating_point() else tensor)
        for key, tensor in state.items()
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(dest))
    del state, out
    return dest


def _probe_images():
    """Visually DISTINCT probes, deliberately not noise.

    The first version of this measurement used upscaled random noise, and reported that
    fp16 changed the ranking. It does - but so would anything, because noise images are all
    roughly equidistant to CLIP, so their pairwise similarities sit within ~1e-5 of each
    other and any perturbation reorders near-ties. That is a property of the probe set, not
    a fact about fp16. Distinct images have real margins, which is the situation the product
    is actually in when it ranks one campaign deck against another.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    size = (224, 224)

    def solid(rgb):
        return lambda: Image.new("RGB", size, rgb)

    def gradient(axis):
        def draw():
            a = np.zeros((size[1], size[0], 3), dtype="uint8")
            ramp = np.linspace(0, 255, size[axis], dtype="uint8")
            a[:, :, 0] = ramp[None, :] if axis == 0 else ramp[:, None]
            return Image.fromarray(a, "RGB")
        return draw

    def stripes(width, vertical=True):
        def draw():
            a = np.zeros((size[1], size[0], 3), dtype="uint8")
            idx = (np.arange(size[0]) // width) % 2
            band = np.where(idx, 255, 0).astype("uint8")
            a[:, :, :] = (band[None, :, None] if vertical else band[:, None, None])
            return Image.fromarray(a, "RGB")
        return draw

    def checker(cell):
        def draw():
            ys, xs = np.mgrid[0:size[1], 0:size[0]]
            a = (((xs // cell) + (ys // cell)) % 2 * 255).astype("uint8")
            return Image.fromarray(np.dstack([a, a, a]), "RGB")
        return draw

    def shape(kind):
        def draw():
            img = Image.new("RGB", size, (250, 250, 250))
            d = ImageDraw.Draw(img)
            if kind == "circle":
                d.ellipse([40, 40, 184, 184], fill=(200, 30, 30))
            elif kind == "square":
                d.rectangle([50, 50, 174, 174], fill=(30, 80, 200))
            else:
                d.polygon([(112, 40), (184, 184), (40, 184)], fill=(20, 150, 70))
            return img
        return draw

    return [solid((10, 10, 10)), solid((240, 240, 240)), solid((200, 20, 20)),
            solid((20, 60, 200)), gradient(0), gradient(1), stripes(8), stripes(24),
            checker(16), shape("circle"), shape("square"), shape("triangle")]



# Real photographs, when the machine has some. Synthetic shapes measure the EASY case -
# a red square against a gradient has margins no perturbation could close, which flatters
# the result. What this product actually does is rank photographs of one brand's shoots
# against each other, so a set of real photos is the measurement that counts.
_PHOTO_DIRS = [
    Path("/System/Library/Desktop Pictures"),          # macOS ships ~150 photographs
    Path("/usr/share/backgrounds"),                    # common on Linux desktops
]


def _probe_paths(work: Path, photo_dir: Path | None = None) -> tuple[list[Path], str]:
    """Real photographs if we can find any, synthetic shapes otherwise."""
    from PIL import Image

    candidates = [photo_dir] if photo_dir else _PHOTO_DIRS
    for directory in candidates:
        if not directory or not directory.is_dir():
            continue
        found = sorted(
            q for ext in ("*.heic", "*.jpg", "*.jpeg", "*.png")
            for q in directory.rglob(ext)
        )[:25]
        if len(found) >= 8:
            converted = []
            for i, src in enumerate(found):
                try:
                    with Image.open(src) as im:
                        dest = work / f"photo_{i}.png"
                        im.convert("RGB").resize((224, 224), Image.BICUBIC).save(dest)
                        converted.append(dest)
                except Exception:
                    continue  # unreadable/unsupported format, skip it
            if len(converted) >= 8:
                return converted, f"{len(converted)} real photographs from {directory}"

    paths = [work / f"probe_{i}.png" for i in range(12)]
    probes = _probe_images()
    for draw, q in zip(probes, paths):
        draw().save(q)
    return paths, f"{len(paths)} synthetic shapes (EASY case - margins are unrealistically wide)"


def measure_agreement(fp32: Path, fp16: Path, photo_dir: Path | None = None) -> dict:
    """Embed the same images under both checkpoints and compare.

    Two things matter, and only one of them is the obvious one. Absolute cosine drift
    between the two embeddings of one image is the easy metric; the one that actually
    decides whether this is shippable is whether the RANKING changes, because ranking is
    the entire product feature ("which prior campaign does this look like")."""
    import shutil

    tmp = Path(fp16).parent / "_agreement_probe"
    shutil.rmtree(tmp, ignore_errors=True)   # a leftover set from a failed run skews nothing
    tmp.mkdir(parents=True)
    try:
        paths, probe_label = _probe_paths(tmp, photo_dir)

        result = _compare(fp32, fp16, paths, probe_label)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return result


def _compare(fp32: Path, fp16: Path, paths, probe_label: str) -> dict:
    import numpy as np
    import open_clip
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config
    import config

    def embed_all(checkpoint: Path):
        model, _, preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=str(checkpoint))
        model.eval()
        vecs = []
        with torch.no_grad():
            for p in paths:
                x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                v = model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
                vecs.append(v.squeeze(0).float().numpy())
        return np.stack(vecs)

    a = embed_all(fp32)
    b = embed_all(fp16)

    per_image_cos = (a * b).sum(axis=1)
    sims_a, sims_b = a @ a.T, b @ b.T
    order_a = np.argsort(-sims_a, axis=1)
    order_b = np.argsort(-sims_b, axis=1)

    # How often does the fp16 perturbation actually exceed the gap between adjacent
    # candidates? That - not the best-case margin - is the probability of a visible reorder.
    tight = sum(1 for row, order in zip(sims_a, order_a)
                for x, y in zip(order[:5], order[1:6])
                if abs(row[x] - row[y]) < np.abs(sims_a - sims_b).max())

    # The gap between each image's 1st and 2nd nearest neighbour (excluding itself). If
    # that margin is orders of magnitude larger than the fp16 perturbation, ranking is
    # stable for any input with real visual differences.
    margins = []
    for row, order in zip(sims_a, order_a):
        ranked = [j for j in order if row[j] < 0.99999]  # drop only the self-match
        if len(ranked) >= 2:
            margins.append(float(row[ranked[0]] - row[ranked[1]]))
    delta = float(np.abs(sims_a - sims_b).max())

    return {
        "probe_set": probe_label,
        "min_cosine_same_image": float(per_image_cos.min()),
        "mean_cosine_same_image": float(per_image_cos.mean()),
        "max_similarity_delta": delta,
        "min_rank_margin": min(margins) if margins else None,
        "margin_over_perturbation": (min(margins) / delta) if margins and delta else None,
        "adjacent_gaps_below_perturbation": tight,
        "similarity_band": f"{sims_a[~np.eye(len(paths), dtype=bool)].min():.3f}"
                           f"-{sims_a[~np.eye(len(paths), dtype=bool)].max():.3f}",
        "top5_order_identical": bool((order_a[:, :5] == order_b[:, :5]).all()),
        "ranking_identical": bool((order_a == order_b).all()),
        "top1_identical": bool((order_a[:, :1] == order_b[:, :1]).all()),
    }


def main() -> int:
    if len(sys.argv) not in (3, 4):
        print(__doc__)
        return 2
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    photo_dir = Path(sys.argv[3]) if len(sys.argv) == 4 else None
    print(f"converting {src} -> {dest}")
    convert(src, dest)
    print(f"fp32: {src.stat().st_size:,} bytes")
    print(f"fp16: {dest.stat().st_size:,} bytes "
          f"({dest.stat().st_size / src.stat().st_size:.1%} of original)")
    print("\nmeasuring agreement ...")
    for key, value in measure_agreement(src, dest, photo_dir).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
