# Очікуваний формат (14 чисел, id знімка береться з імені файлу `00000000.txt`):
# `x_center y_center angle(рад) class_id is_contained is_occluded x1 x2 x3 x4 y1 y2 y3 y4`
# Рядки іншого формату: рахуються, показуються в діагностиці `inspect_raw()`,
# обробляються за `bad_line_policy`

from PIL import Image, ImageDraw, ImageFont
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import csv, json, math, random, shutil, statistics
import yaml

DATA_CFG = {} # взяти з configs.data.vedai

# при запуску вказати шляхи
RAW_ROOT = Path("")                    # розпакований VEDAI
DATA_DIR = Path("")                   # датасет в форматі YOLO-OBB (локально)
RUNS_DIR = Path("")


IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
           "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080", "#9a6324"]


def show(path):
    try:
        from IPython.display import display
        display(Image.open(path))
    except Exception:
        print("Збережено:", path)


def get_font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def norm_id(s):
    s = Path(str(s)).stem
    for suf in ("_co", "_ir"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    return s.zfill(8) if s.isdigit() else s


def raw_paths(cfg=DATA_CFG):
    """Папка зі знімками та папка з анотаціями (з конфігу або автопошуком)"""
    img_dir, ann_dir = cfg["images_dir"], cfg["annotations"]
    if img_dir is None or ann_dir is None:
        imgs = Counter(p.parent for p in RAW_ROOT.rglob(f"*{cfg['rgb_suffix']}.*") if p.suffix.lower() in IMG_EXT)
        anns = Counter(p.parent for p in RAW_ROOT.rglob("*.txt") if p.stem.isdigit())

        def pick(c):
            pool = [d for d in c if "1024" in str(d)] or list(c)
            return max(pool, key=lambda d: c[d]) if pool else None
        img_dir = img_dir or pick(imgs)
        ann_dir = ann_dir or pick(anns)
    if img_dir is None or ann_dir is None:
        raise SystemExit(f"Не знайдено знімки/анотації в {RAW_ROOT}")
    return Path(img_dir), Path(ann_dir)


def _floats(line):
    try:
        return [float(x) for x in line.split()]
    except ValueError:
        return []


def parse_full_line(line, img_id):
    v = _floats(line)
    if len(v) != 14:
        return None
    return {"img_id": img_id, "xc": v[0], "yc": v[1], "angle": v[2], "cls": int(v[3]),
            "contained": int(v[4]), "occluded": int(v[5]), "xs": v[6:10], "ys": v[10:14], "raw": line.strip()}


def load_annotations(ann_dir, tol=3.0):
    """-> (ann, bad).
    ann: {img_id: [повні об'єкти]};
    bad: рядки не у форматі з 14 чисел.
    matched=True, якщо центр такого рядка збігається з центром якогось повного об'єкта (дубль)"""
    files = [p for p in sorted(Path(ann_dir).rglob("*.txt")) if p.stem.isdigit()]
    if not files:
        raise SystemExit(f"У {ann_dir} немає файлів 00000000.txt")
    ann, bad = {}, []
    for f in files:
        img_id = norm_id(f.stem)
        full, short = [], []
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            rec = parse_full_line(line, img_id)
            (full if rec else short).append(rec or line.strip())
        ann[img_id] = full
        for line in short:
            v = _floats(line)
            matched = len(v) >= 2 and any(abs(v[0] - o["xc"]) <= tol and abs(v[1] - o["yc"]) <= tol for o in full)
            bad.append({"img_id": img_id, "file": f.name, "n_tokens": len(line.split()), "line": line, "matched": matched})
    return ann, bad


def list_images(images_dir, rgb_suffix):
    return {norm_id(p.stem): p for p in sorted(Path(images_dir).rglob("*"))
            if p.suffix.lower() in IMG_EXT and p.stem.endswith(rgb_suffix)}


def report_bad_lines(bad, ann_dir, n_files):
    if not bad:
        print("Рядків не у форматі з 14 чисел: 0")
        return
    files = {b["file"] for b in bad}
    print(f"\n!!! Рядків не у форматі з 14 чисел: {len(bad)} у {len(files)} з {n_files} файлів")
    for n_tok, cnt in sorted(Counter(b["n_tokens"] for b in bad).items()):
        ex = [b for b in bad if b["n_tokens"] == n_tok][:3]
        dup = sum(b["matched"] for b in bad if b["n_tokens"] == n_tok)
        print(f"  {n_tok} чисел у рядку: {cnt} рядків, з них дублюють центр повного об'єкта: {dup}")
        for b in ex:
            print(f"     {b['file']}: {b['line']!r}")
    unmatched = [b for b in bad if not b["matched"]]
    print(f"Рядків, що НЕ дублюють жодного повного об'єкта: {len(unmatched)} у {len({b['img_id'] for b in unmatched})} знімках")
    first = bad[0]["file"]
    print(f"\nВміст першого проблемного файлу {first}:")
    print((Path(ann_dir) / first).read_text()[:1200])


def peek_images(ids, cfg=DATA_CFG, n=6, size=380):
    """Показати кілька знімків за id (наприклад, ті, що без файлу анотацій)."""
    img_dir, _ = raw_paths(cfg)
    images = list_images(img_dir, cfg["rgb_suffix"])
    ids = sorted(ids)[:n]
    cols = min(3, len(ids))
    sheet = Image.new("RGB", (cols * size, math.ceil(len(ids) / cols) * size), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for j, i in enumerate(ids):
        sheet.paste(Image.open(images[i]).convert("RGB").resize((size, size)), ((j % cols) * size, (j // cols) * size))
        dr.text(((j % cols) * size + 6, (j // cols) * size + 4), i, fill="#ffe119", font=get_font(18))
    path = DATA_DIR / "viz" / "no_annotation_examples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    show(path)


def plot_counts(path, overall, images_with=None, split_counts=None, split_title=""):
    order = sorted(overall, key=lambda c: -overall[c])
    n_panels = 1 + int(split_counts is not None or images_with is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 0.45 * len(order) + 2.2))
    axes = [axes] if n_panels == 1 else list(axes)
    bars = axes[0].barh(order, [overall[c] for c in order], color="#4363d8")
    axes[0].invert_yaxis(); axes[0].set_xlabel("кількість об'єктів")
    axes[0].set_title(f"Об'єкти по класах (усього {sum(overall.values())})")
    axes[0].bar_label(bars, padding=3); axes[0].margins(x=0.12)
    if split_counts is not None:
        ax, left = axes[1], [0] * len(order)
        for part, col in zip(("train", "val", "test"), ("#4363d8", "#f58231", "#3cb44b")):
            vals = [split_counts[part].get(c, 0) for c in order]
            ax.barh(order, vals, left=left, color=col, label=part)
            for i, (v, l) in enumerate(zip(vals, left)):
                if v > 0:
                    ax.text(l + v / 2, i, str(v), ha="center", va="center", fontsize=8, color="white")
            left = [a + b for a, b in zip(left, vals)]
        ax.invert_yaxis(); ax.set_title(split_title or "train / val / test"); ax.legend()
    elif images_with is not None:
        bars = axes[1].barh(order, [images_with.get(c, 0) for c in order], color="#3cb44b")
        axes[1].invert_yaxis(); axes[1].set_title("Знімків, де клас зустрічається")
        axes[1].bar_label(bars, padding=3); axes[1].margins(x=0.12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def class_gallery(ann, images, classes, k, path, seed=0, win=96, scale=3):
    """Рядок на клас: підпис + k кропів з накладеною рамкою."""
    by_cls = defaultdict(list)
    for recs in ann.values():
        for o in recs:
            by_cls[o["cls"]].append(o)
    ids = list(classes) + sorted(set(by_cls) - set(classes))
    rng, cell, label_w, font = random.Random(seed), win * scale, 230, get_font(20)
    sheet = Image.new("RGB", (label_w + k * cell, len(ids) * cell), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for r, cid in enumerate(ids):
        recs = [o for o in by_cls.get(cid, []) if o["img_id"] in images]
        dr.text((10, r * cell + cell // 2 - 24), f"id {cid}", fill="white", font=font)
        dr.text((10, r * cell + cell // 2), f"{classes.get(cid, 'НЕ В КОНФІГУ')}  (n={len(by_cls.get(cid, []))})",
                fill="#ffe119", font=font)
        for j, o in enumerate(rng.sample(recs, min(k, len(recs)))):
            im = Image.open(images[o["img_id"]]).convert("RGB")
            cx, cy = int(o["xc"]), int(o["yc"])
            box = (cx - win // 2, cy - win // 2, cx + win // 2, cy + win // 2)
            crop = im.crop(box).resize((cell, cell), Image.LANCZOS)
            poly = [((x - box[0]) * scale, (y - box[1]) * scale) for x, y in zip(o["xs"], o["ys"])]
            ImageDraw.Draw(crop).polygon(poly, outline="#ffe119", width=2)
            sheet.paste(crop, (label_w + j * cell, r * cell))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def inspect_raw(cfg=DATA_CFG, examples=3):
    """Діагностика сирих даних: id класів, по прикладу кожного класу, формат, погані рядки, графіки."""
    classes, tol = cfg["classes"], cfg["centroid_tol_px"]
    img_dir, ann_dir = raw_paths(cfg)
    print("Знімки:", img_dir, "\nАнотації:", ann_dir)
    ann, bad = load_annotations(ann_dir, tol)
    images = list_images(img_dir, cfg["rgb_suffix"])
    n_obj = sum(len(v) for v in ann.values())
    print(f"\nФайлів анотацій: {len(ann)} | RGB-знімків: {len(images)} | повних об'єктів (14 чисел): {n_obj}")
    print("Анотації без знімка:", len(set(ann) - set(images)), "| знімки без файлу анотацій:", len(set(images) - set(ann)))
    report_bad_lines(bad, ann_dir, len(ann))
    no_ann = sorted(set(images) - set(ann))
    if no_ann:
        print(f"\nЗнімки БЕЗ файлу анотацій ({len(no_ann)}): {no_ann[:20]}")
        print("Подивіться, чи є на них транспорт (no_annotation_examples.png). Якщо є, їх не можна брати як 'фон'.")
        peek_images(no_ann, cfg)
    excl = {c for v in ann.values() for o in v for c in [o["cls"]] if c not in classes}
    if excl:
        n_img = len({o["img_id"] for v in ann.values() for o in v if o["cls"] not in classes})
        print(f"\nКласи поза конфігом {sorted(excl)}: знімків із такими об'єктами {n_img} (політика: {cfg['excluded_class_policy']})")

    counts = Counter(o["cls"] for v in ann.values() for o in v)
    img_counts = Counter(c for v in ann.values() for c in {o["cls"] for o in v})
    print("\nid -> об'єктів / знімків (назва з конфігу):")
    for cid, n in sorted(counts.items()):
        print(f"  {cid:>3}: {n:>5} / {img_counts[cid]:>4}   {classes.get(cid, 'НЕ В КОНФІГУ (буде відкинуто)')}")
    print("Класи з конфігу, яких немає в даних:", [c for c in classes if c not in counts] or "немає")

    print("\nПо одному прикладу кожного класу (id -> назва -> індекс YOLO):")
    idx, rng = {cid: i for i, cid in enumerate(classes)}, random.Random(0)
    for cid in sorted(counts):
        o = rng.choice([o for v in ann.values() for o in v if o["cls"] == cid])
        print(f"  id {cid:>2} -> {classes.get(cid, '???'):<12} idx={idx.get(cid, '-')}  знімок {o['img_id']}")
        print(f"       рядок: {o['raw']}")

    bad_c = sum(abs(sum(o["xs"]) / 4 - o["xc"]) > tol or abs(sum(o["ys"]) / 4 - o["yc"]) > tol
                for v in ann.values() for o in v)
    maxc = max((max(*o["xs"], *o["ys"]) for v in ann.values() for o in v), default=0)
    print(f"\nПеревірка формату: {bad_c}/{n_obj} об'єктів, де центр != центр чотирьох кутів (допуск {tol}px)")
    if images:
        w, h = Image.open(next(iter(images.values()))).size
        print(f"Макс. координата: {maxc:.0f}, розмір знімка: {w}x{h}")

    out = DATA_DIR / "viz"
    nm = {c: classes.get(c, f"id{c}?") for c in counts}
    plot_counts(out / "class_counts_raw.png", {nm[c]: n for c, n in counts.items()},
                images_with={nm[c]: n for c, n in img_counts.items()})
    class_gallery(ann, images, classes, examples, out / "class_examples.png")
    show(out / "class_counts_raw.png")
    show(out / "class_examples.png")
    print("Подивіться class_examples.png: у кожному рядку мають бути об'єкти, що відповідають назві класу.")


def poly_sides(pts):
    s1, s2 = math.dist(pts[0], pts[1]), math.dist(pts[1], pts[2])
    return max(s1, s2), min(s1, s2)


def _stratum(per_image, ids):
    """Страта знімка = його найрідкісніший клас (за кількістю знімків із цим класом); фон = '_background'."""
    freq = Counter(c for i in ids for c in per_image[i])
    return {i: (min(per_image[i], key=lambda c: (freq[c], c)) if per_image[i] else "_background") for i in ids}


def _spread(strata_groups, n, rng):
    """Round-robin розкладка по n кошиках, від найрідкіснішої страти; зсув зберігає однакові розміри кошиків."""
    buckets, offset = [[] for _ in range(n)], 0
    for s_name in sorted(strata_groups, key=lambda k: (len(strata_groups[k]), str(k))):
        lst = sorted(strata_groups[s_name])
        rng.shuffle(lst)
        for j, i in enumerate(lst):
            buckets[(offset + j) % n].append(i)
        offset += len(lst)
    return buckets


def make_folds(ids, scfg, per_image=None):
    """{fold: {train, val, test}}. stratify=True (потрібен per_image): рідкісні класи розподіляються по фолдах рівно."""
    n_folds, seed, off = scfg["n_folds"], scfg["seed"], scfg.get("official_folds_dir")
    strat = bool(scfg.get("stratify")) and per_image is not None and not off
    id_set, folds = set(ids), {}
    strata = _stratum(per_image, ids) if strat else {i: "all" for i in ids}
    if off:
        def read(p):
            return [norm_id(l.split()[0]) for l in Path(p).read_text().splitlines() if l.strip()]
        for k in range(1, n_folds + 1):
            pool = [i for i in read(Path(off) / f"fold{k:02d}.txt") if i in id_set]
            test = [i for i in read(Path(off) / f"fold{k:02d}test.txt") if i in id_set]
            assert not (set(pool) & set(test)), f"fold {k}: train-пул і test перетинаються"
            folds[k] = {"pool": pool, "test": test}
    else:
        groups = defaultdict(list)
        for i in ids:
            groups[strata[i]].append(i)
        chunks = _spread(groups, n_folds, random.Random(seed))
        for k in range(1, n_folds + 1):
            folds[k] = {"test": chunks[k - 1], "pool": [i for j, c in enumerate(chunks) if j != k - 1 for i in c]}
    for k, f in folds.items():
        rng = random.Random(seed * 1000 + k)
        by_s = defaultdict(list)
        for i in f["pool"]:
            by_s[strata[i]].append(i)
        val = []
        for lst in by_s.values():                      # val береться з кожної страти пропорційно
            lst = sorted(lst)
            rng.shuffle(lst)
            n_val = round(scfg["val_fraction"] * len(lst))
            val += lst[: max(n_val, 1) if len(lst) >= 2 else n_val]
        f["val"] = sorted(val)
        f["train"] = sorted(set(f["pool"]) - set(val))
        f["test"] = sorted(f["test"])
        del f["pool"]
    return folds


def convert_dataset(cfg=DATA_CFG, plot_fold=1):
    """Сирі дані -> DATA_DIR: images/, labels/, meta/, foldNN_{train,val,test}.txt, vedai_foldNN.yaml."""
    out, classes = DATA_DIR, cfg["classes"]
    cls_index = {cid: i for i, cid in enumerate(classes)}
    names = {i: classes[cid] for cid, i in cls_index.items()}
    tol, min_side = cfg["centroid_tol_px"], cfg["min_side_px"]
    img_dir, ann_dir = raw_paths(cfg)
    ann, bad = load_annotations(ann_dir, tol)
    images = list_images(img_dir, cfg["rgb_suffix"])
    if not images:
        raise SystemExit(f"Не знайдено знімків із суфіксом {cfg['rgb_suffix']} у {img_dir}")

    drops = {"bad_lines": set(), "no_annotation_file": set(), "excluded_class": set()}
    if cfg["bad_line_policy"] == "drop_image":
        drops["bad_lines"] = {b["img_id"] for b in bad if not b["matched"]}
    if cfg["missing_ann_policy"] == "drop_image":
        drops["no_annotation_file"] = set(images) - set(ann)
    if cfg["excluded_class_policy"] == "drop_image":
        drops["excluded_class"] = {i for i, recs in ann.items() if any(o["cls"] not in cls_index for o in recs)}
    drop_ids = set().union(*drops.values())
    report_bad_lines(bad, ann_dir, len(ann))
    print("Викинуто знімків:", {k: len(v) for k, v in drops.items()}, "| разом унікальних:", len(drop_ids))

    for sub in ("images", "labels", "meta", "viz"):          # чистимо старі результати
        shutil.rmtree(out / sub, ignore_errors=True)
        (out / sub).mkdir(parents=True, exist_ok=True)
    for p in list(out.glob("fold*_*.txt")) + list(out.glob("vedai_fold*.yaml")) + [out / "labels.cache"]:
        p.unlink(missing_ok=True)

    dropped, unknown_ids, sizes = Counter(), Counter(), defaultdict(list)
    per_image, meta_rows, n_empty, centroid_bad, max_ratio = {}, [], 0, 0, 0.0
    kept_images = {i: p for i, p in images.items() if i not in drop_ids}

    for img_id, img_path in sorted(kept_images.items()):
        w, h = Image.open(img_path).size
        dst = out / "images" / f"{img_id}.png"
        if img_path.suffix.lower() == ".png":
            shutil.copy2(img_path, dst)
        else:
            Image.open(img_path).convert("RGB").save(dst)
        lines, per_image[img_id] = [], Counter()
        for o in ann.get(img_id, []):
            if o["cls"] not in cls_index:
                dropped["unknown_class"] += 1
                unknown_ids[o["cls"]] += 1
                continue
            if abs(sum(o["xs"]) / 4 - o["xc"]) > tol or abs(sum(o["ys"]) / 4 - o["yc"]) > tol:
                centroid_bad += 1
            max_ratio = max(max_ratio, max(o["xs"]) / w, max(o["ys"]) / h)
            cx, cy = sum(o["xs"]) / 4, sum(o["ys"]) / 4
            if not (0 <= cx < w and 0 <= cy < h):
                dropped["center_outside"] += 1
                continue
            pts = [(min(max(x, 0.0), w), min(max(y, 0.0), h)) for x, y in zip(o["xs"], o["ys"])]
            long_s, short_s = poly_sides(pts)
            if short_s < min_side:
                dropped["degenerate"] += 1
                continue
            if any(x < 0 or x > w or y < 0 or y > h for x, y in zip(o["xs"], o["ys"])):
                dropped["clipped_to_border(kept)"] += 1
            ci = cls_index[o["cls"]]
            lines.append(f"{ci} " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in pts))
            sizes[names[ci]].append(long_s)
            per_image[img_id][names[ci]] += 1
            meta_rows.append({"image_id": img_id, "obj_idx": len(lines) - 1, "orig_class_id": o["cls"],
                              "class_idx": ci, "class_name": names[ci], "occluded": o["occluded"],
                              "contained": o["contained"], "long_side_px": round(long_s, 2),
                              "short_side_px": round(short_s, 2), "angle_rad": o["angle"]})
        n_empty += not lines
        (out / "labels" / f"{img_id}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    if not meta_rows:
        raise SystemExit("Жодного об'єкта не збережено: перевірте класи й формат (inspect_raw).")

    with open(out / "meta" / "objects.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(meta_rows[0]))
        wr.writeheader(); wr.writerows(meta_rows)

    folds = make_folds(sorted(kept_images), cfg["splits"], per_image)
    split_counts = {}
    for k, f in folds.items():
        split_counts[k] = {}
        for part in ("train", "val", "test"):
            (out / f"fold{k:02d}_{part}.txt").write_text("\n".join(f"./images/{i}.png" for i in f[part]) + "\n")
            c = Counter()
            for i in f[part]:
                c.update(per_image[i])
            split_counts[k][part] = dict(c)
        (out / f"vedai_fold{k:02d}.yaml").write_text(yaml.safe_dump(
            {"train": f"fold{k:02d}_train.txt", "val": f"fold{k:02d}_val.txt",
             "test": f"fold{k:02d}_test.txt", "names": names}, sort_keys=False, allow_unicode=True))

    overall = Counter(r["class_name"] for r in meta_rows)
    missing = {k: {p: [n for n in names.values() if split_counts[k][p].get(n, 0) == 0] for p in ("val", "test")}
               for k in folds}
    stats = {"n_images_kept": len(kept_images), "n_images_dropped": {k: len(v) for k, v in drops.items()},
             "n_images_without_objects": n_empty, "n_objects_kept": len(meta_rows),
             "bad_lines_total": len(bad), "bad_lines_unmatched": sum(not b["matched"] for b in bad),
             "objects_per_class": dict(overall), "dropped": dict(dropped), "unknown_class_ids": dict(unknown_ids),
             "centroid_mismatch_objects": centroid_bad, "max_coord_over_image_size": round(max_ratio, 3),
             "long_side_px": {c: {"median": round(statistics.median(v), 1), "min": round(min(v), 1),
                                  "max": round(max(v), 1)} for c, v in sizes.items()},
             "folds": {k: {p: len(f[p]) for p in ("train", "val", "test")} for k, f in folds.items()},
             "class_counts_per_split": split_counts, "classes_missing_in_split": missing}
    (out / "meta" / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    plot_counts(out / "meta" / "class_counts.png", dict(overall), split_counts=split_counts[plot_fold],
                split_title=f"fold{plot_fold:02d}: train / val / test")
    print(json.dumps({k: stats[k] for k in ("n_images_kept", "n_images_dropped", "n_images_without_objects",
                                            "n_objects_kept", "objects_per_class", "dropped", "unknown_class_ids")},
                     indent=2, ensure_ascii=False))
    bad_folds = {k for k, v in missing.items() if v["val"] or v["test"]}
    if bad_folds:
        print(f"\nУВАГА: у {len(bad_folds)} з {len(folds)} фолдів є класи без жодного об'єкта у val/test "
              f"(AP для них не визначений). Деталі: meta/stats.json -> classes_missing_in_split.")
    if centroid_bad > 0.05 * len(meta_rows):
        print("УВАГА: багато об'єктів із розбіжністю центру; перевірте формат (inspect_raw).")
    show(out / "meta" / "class_counts.png")
    return stats


def viz_labels(n=12, cfg=DATA_CFG):
    """Рамки з готових labels на випадкових знімках: перевірка, що розмітка лягла на об'єкти."""
    out, names = DATA_DIR, dict(enumerate(cfg["classes"].values()))
    stems = sorted(p.stem for p in (out / "labels").glob("*.txt") if p.stat().st_size > 0)
    random.Random(0).shuffle(stems)
    tiles = []
    for s in stems[:n]:
        im = Image.open(out / "images" / f"{s}.png").convert("RGB")
        w, h = im.size
        dr = ImageDraw.Draw(im)
        for line in (out / "labels" / f"{s}.txt").read_text().splitlines():
            v = line.split()
            ci, pts = int(v[0]), [float(x) for x in v[1:]]
            poly = [(pts[i] * w, pts[i + 1] * h) for i in range(0, 8, 2)]
            col = PALETTE[ci % len(PALETTE)]
            dr.polygon(poly, outline=col, width=2)
            dr.text((poly[0][0] + 2, poly[0][1] + 2), names.get(ci, str(ci)), fill=col)
        tiles.append(im.resize((512, 512)))
    cols = 4
    grid = Image.new("RGB", (cols * 512, math.ceil(len(tiles) / cols) * 512))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * 512, (i // cols) * 512))
    grid.save(out / "viz" / "label_check.png")
    show(out / "viz" / "label_check.png")
