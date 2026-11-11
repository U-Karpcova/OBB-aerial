#!/usr/bin/env python3
"""VEDAI (RGB, 1024x1024) -> Ultralytics YOLO-OBB.

Підкоманди:
  inspect  - подивитись на сирі анотації, id класів, перевірити припущення про формат
  convert  - зібрати images/ labels/, meta/objects.csv, stats.json, списки фолдів і yaml
  viz      - намалювати рамки з отриманих labels на випадкових знімках (візуальна перевірка)

Використання:
  python scripts/prepare_vedai.py inspect --config configs/data/vedai.yaml
  python scripts/prepare_vedai.py convert --config configs/data/vedai.yaml
  python scripts/prepare_vedai.py viz     --config configs/data/vedai.yaml -n 12

Припущений формат рядка анотації VEDAI (14 токенів, або 15, якщо першим іде id знімка):
  [img_id] x_center y_center angle class_id fully_contained occluded x1 x2 x3 x4 y1 y2 y3 y4
Скрипт сам перевіряє це припущення (центр у файлі має збігатися з центром кутів).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


# ----------------------------------------------------------------------------
# Читання сирих анотацій
# ----------------------------------------------------------------------------
def norm_id(s: str) -> str:
    s = Path(str(s)).stem
    for suf in ("_co", "_ir"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.zfill(8) if s.isdigit() else s


def parse_line(line: str, default_id: str | None = None) -> dict:
    t = line.split()
    if len(t) == 15:
        img_id, t = t[0], t[1:]
    elif len(t) == 14:
        img_id = default_id
    else:
        raise ValueError(f"очікую 14 або 15 токенів, отримала {len(t)}: {line!r}")
    if img_id is None:
        raise ValueError("немає id знімка (ні в рядку, ні в імені файлу)")
    return {
        "img_id": norm_id(img_id),
        "xc": float(t[0]),
        "yc": float(t[1]),
        "angle": float(t[2]),
        "cls": int(float(t[3])),
        "contained": int(float(t[4])),
        "occluded": int(float(t[5])),
        "xs": [float(v) for v in t[6:10]],
        "ys": [float(v) for v in t[10:14]],
    }


def find_annotation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    combined = sorted(path.rglob("annotation*.txt"))
    if len(combined) > 1:
        raise SystemExit(
            f"У {path} кілька файлів annotation*.txt: {[c.name for c in combined]}. "
            "Вкажіть у конфігу конкретний файл (annotation1024.txt)."
        )
    if combined:
        return combined
    return [p for p in sorted(path.rglob("*.txt")) if not p.stem.lower().startswith("fold")]


def load_annotations(path: Path) -> dict[str, list[dict]]:
    ann: dict[str, list[dict]] = defaultdict(list)
    for f in find_annotation_files(path):
        for line in f.read_text().splitlines():
            if line.strip():
                rec = parse_line(line, default_id=f.stem)
                ann[rec["img_id"]].append(rec)
    return ann


def list_images(images_dir: Path, rgb_suffix: str) -> dict[str, Path]:
    out = {}
    for p in sorted(images_dir.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.stem.endswith(rgb_suffix):
            out[norm_id(p.stem)] = p
    return out


# ----------------------------------------------------------------------------
# inspect
# ----------------------------------------------------------------------------
def cmd_inspect(cfg: dict) -> None:
    raw = cfg["raw"]
    files = find_annotation_files(Path(raw["annotations"]))
    print(f"Файли анотацій: {len(files)} (перший: {files[0]})")
    print("Перші рядки:")
    for line in files[0].read_text().splitlines()[:3]:
        print("  ", line)

    ann = load_annotations(Path(raw["annotations"]))
    images = list_images(Path(raw["images_dir"]), raw["rgb_suffix"])
    n_obj = sum(len(v) for v in ann.values())
    print(f"\nЗнімків з анотаціями: {len(ann)}, RGB-знімків на диску: {len(images)}, об'єктів: {n_obj}")
    print("Анотації без знімка:", len(set(ann) - set(images)), "| знімки без анотацій:", len(set(images) - set(ann)))

    counts = Counter(o["cls"] for v in ann.values() for o in v)
    known = cfg["classes"]
    print("\nid класу -> кількість (у конфігу? назва):")
    for cid, n in sorted(counts.items()):
        print(f"  {cid:>3}: {n:>5}   {'так: ' + known[cid] if cid in known else 'НІ (буде відкинуто)'}")
    missing = [c for c in known if c not in counts]
    if missing:
        print("Класи з конфігу, яких немає в даних:", missing)

    # Перевірка припущення про порядок колонок
    tol = cfg.get("centroid_tol_px", 3)
    bad, maxcoord = 0, 0.0
    for v in ann.values():
        for o in v:
            cx, cy = sum(o["xs"]) / 4, sum(o["ys"]) / 4
            if abs(cx - o["xc"]) > tol or abs(cy - o["yc"]) > tol:
                bad += 1
            maxcoord = max(maxcoord, *o["xs"], *o["ys"])
    print(f"\nПеревірка формату: {bad}/{n_obj} об'єктів, де центр у файлі != центр кутів (допуск {tol}px)")
    print(f"Максимальна координата в анотаціях: {maxcoord:.1f}")
    if images:
        w, h = Image.open(next(iter(images.values()))).size
        print(f"Розмір першого знімка: {w}x{h}")
        if maxcoord < 0.6 * w:
            print("УВАГА: координати виглядають меншими за розмір знімка (анотації для 512 при знімках 1024?)")
    if n_obj and bad / n_obj > 0.05:
        print("УВАГА: припущення про порядок колонок, схоже, хибне; перегляньте devkit/readme.")


# ----------------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------------
def poly_sides(pts: list[tuple[float, float]]) -> tuple[float, float]:
    s1 = math.dist(pts[0], pts[1])
    s2 = math.dist(pts[1], pts[2])
    return max(s1, s2), min(s1, s2)


def read_official_ids(path: Path) -> list[str]:
    ids = []
    for line in path.read_text().splitlines():
        if line.strip():
            ids.append(norm_id(line.split()[0]))
    return ids


def make_folds(ids: list[str], scfg: dict) -> dict[int, dict[str, list[str]]]:
    """Повертає {fold: {train, val, test}} зі списками id."""
    n_folds, seed = scfg["n_folds"], scfg["seed"]
    off = scfg.get("official_folds_dir")
    folds: dict[int, dict[str, list[str]]] = {}
    id_set = set(ids)

    if off:
        off = Path(off)
        for k in range(1, n_folds + 1):
            pool = [i for i in read_official_ids(off / f"fold{k:02d}.txt") if i in id_set]
            test = [i for i in read_official_ids(off / f"fold{k:02d}test.txt") if i in id_set]
            assert not (set(pool) & set(test)), f"fold {k}: train-пул і test перетинаються"
            folds[k] = {"pool": pool, "test": test}
    else:
        shuffled = sorted(ids)
        random.Random(seed).shuffle(shuffled)
        chunks = [shuffled[i::n_folds] for i in range(n_folds)]
        for k in range(1, n_folds + 1):
            test = chunks[k - 1]
            pool = [i for j, c in enumerate(chunks) for i in c if j != k - 1]
            folds[k] = {"pool": pool, "test": test}

    for k, f in folds.items():
        pool = sorted(f["pool"])
        random.Random(seed * 1000 + k).shuffle(pool)
        n_val = max(1, round(scfg["val_fraction"] * len(pool)))
        f["val"], f["train"] = sorted(pool[:n_val]), sorted(pool[n_val:])
        f["test"] = sorted(f["test"])
        del f["pool"]
    return folds


def cmd_convert(cfg: dict) -> None:
    raw, out = cfg["raw"], Path(cfg["out_dir"])
    classes: dict[int, str] = cfg["classes"]
    cls_index = {cid: i for i, cid in enumerate(classes)}
    names = {i: classes[cid] for cid, i in cls_index.items()}
    tol, min_side = cfg.get("centroid_tol_px", 3), cfg.get("min_side_px", 2)

    ann = load_annotations(Path(raw["annotations"]))
    images = list_images(Path(raw["images_dir"]), raw["rgb_suffix"])
    if not images:
        raise SystemExit(f"Не знайдено знімків з суфіксом {raw['rgb_suffix']} у {raw['images_dir']}")

    for sub in ("images", "labels", "meta", "viz"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    dropped = Counter()
    unknown_ids = Counter()
    centroid_bad = 0
    per_class_sizes: dict[str, list[float]] = defaultdict(list)
    meta_rows = []
    n_empty = 0
    max_coord_ratio = 0.0

    for img_id, img_path in sorted(images.items()):
        w, h = Image.open(img_path).size
        dst_img = out / "images" / f"{img_id}.png"
        if not dst_img.exists():
            if img_path.suffix.lower() == ".png":
                shutil.copy2(img_path, dst_img)
            else:
                Image.open(img_path).convert("RGB").save(dst_img)

        lines = []
        for o in ann.get(img_id, []):
            if o["cls"] not in cls_index:
                dropped["unknown_class"] += 1
                unknown_ids[o["cls"]] += 1
                continue
            if abs(sum(o["xs"]) / 4 - o["xc"]) > tol or abs(sum(o["ys"]) / 4 - o["yc"]) > tol:
                centroid_bad += 1
            max_coord_ratio = max(max_coord_ratio, max(o["xs"]) / w, max(o["ys"]) / h)

            cx, cy = sum(o["xs"]) / 4, sum(o["ys"]) / 4
            if not (0 <= cx < w and 0 <= cy < h):
                dropped["center_outside"] += 1
                continue
            pts_px = [(min(max(x, 0.0), w), min(max(y, 0.0), h)) for x, y in zip(o["xs"], o["ys"])]
            long_s, short_s = poly_sides(pts_px)
            if short_s < min_side:
                dropped["degenerate"] += 1
                continue
            if any(x < 0 or x > w or y < 0 or y > h for x, y in zip(o["xs"], o["ys"])):
                dropped["clipped_to_border(kept)"] += 1

            ci = cls_index[o["cls"]]
            coords = " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts_px)
            lines.append(f"{ci} {coords}")
            per_class_sizes[names[ci]].append(long_s)
            meta_rows.append({
                "image_id": img_id, "obj_idx": len(lines) - 1, "orig_class_id": o["cls"],
                "class_idx": ci, "class_name": names[ci], "occluded": o["occluded"],
                "contained": o["contained"], "long_side_px": round(long_s, 2),
                "short_side_px": round(short_s, 2), "angle_raw": o["angle"],
            })
        if not lines:
            n_empty += 1
        (out / "labels" / f"{img_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    orphan = sorted(set(ann) - set(images))

    with open(out / "meta" / "objects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
        wr.writeheader()
        wr.writerows(meta_rows)

    # --- розбиття ---
    ids = sorted(images)
    folds = make_folds(ids, cfg["splits"])
    for k, f in folds.items():
        for part in ("train", "val", "test"):
            (out / f"fold{k:02d}_{part}.txt").write_text("\n".join(f"./images/{i}.png" for i in f[part]) + "\n")
        data_yaml = {
            "train": f"fold{k:02d}_train.txt",
            "val": f"fold{k:02d}_val.txt",
            "test": f"fold{k:02d}_test.txt",
            "names": names,
        }
        (out / f"vedai_fold{k:02d}.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    # --- статистика ---
    obj_per_class = Counter(r["class_name"] for r in meta_rows)
    stats = {
        "n_images": len(images),
        "n_images_without_objects": n_empty,
        "n_objects_kept": len(meta_rows),
        "objects_per_class": dict(obj_per_class),
        "dropped": dict(dropped),
        "unknown_class_ids": dict(unknown_ids),
        "annotation_ids_without_image": orphan[:20],
        "centroid_mismatch_objects": centroid_bad,
        "max_coord_over_image_size": round(max_coord_ratio, 3),
        "long_side_px": {
            c: {"median": round(statistics.median(v), 1), "min": round(min(v), 1), "max": round(max(v), 1)}
            for c, v in per_class_sizes.items()
        },
        "folds": {k: {p: len(f[p]) for p in ("train", "val", "test")} for k, f in folds.items()},
    }
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if centroid_bad > 0.05 * max(1, len(meta_rows)):
        print("\nУВАГА: багато об'єктів з розбіжністю центру; перевірте порядок колонок (див. `inspect`, `viz`).",
              file=sys.stderr)
    if max_coord_ratio > 1.05:
        print("\nУВАГА: координати виходять за межі знімка; можливо, анотації для іншої роздільної здатності.",
              file=sys.stderr)


# ----------------------------------------------------------------------------
# viz
# ----------------------------------------------------------------------------
def cmd_viz(cfg: dict, n: int, seed: int = 0) -> None:
    out = Path(cfg["out_dir"])
    names = {i: n_ for i, n_ in enumerate(cfg["classes"].values())}
    palette = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
               "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe"]
    stems = sorted(p.stem for p in (out / "labels").glob("*.txt") if p.stat().st_size > 0)
    random.Random(seed).shuffle(stems)
    tiles = []
    for s in stems[:n]:
        im = Image.open(out / "images" / f"{s}.png").convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for line in (out / "labels" / f"{s}.txt").read_text().splitlines():
            v = line.split()
            ci, pts = int(v[0]), [float(x) for x in v[1:]]
            poly = [(pts[i] * w, pts[i + 1] * h) for i in range(0, 8, 2)]
            col = palette[ci % len(palette)]
            dr.polygon(poly, outline=col, width=2)
            dr.text((poly[0][0] + 2, poly[0][1] + 2), names.get(ci, str(ci)), fill=col)
        tiles.append(im.resize((512, 512)))
    cols = 4
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * 512, rows * 512))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * 512, (i // cols) * 512))
    path = out / "viz" / "label_check.png"
    grid.save(path)
    print("Збережено:", path)


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["inspect", "convert", "viz"])
    ap.add_argument("--config", default="configs/data/vedai.yaml")
    ap.add_argument("-n", type=int, default=12, help="скільки знімків малювати для viz")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.cmd == "inspect":
        cmd_inspect(cfg)
    elif args.cmd == "convert":
        cmd_convert(cfg)
    else:
        cmd_viz(cfg, args.n)


if __name__ == "__main__":
    main()
