"""Демо OBB-aerial: орієнтовані рамки для техніки на аерознімках (YOLO26-OBB, VEDAI).

Запуск:  streamlit run app.py   (потрібен streamlit >= 1.40)
Моделі:  усі best.pt з теки прогонів (runs/<датасет>/yolo26<n|s|m>/<init>/foldNN_seedS/<етап>/weights/best.pt),
         офіційні DOTA-ваги (zero-shot) або власний .pt файл.
Працює на CPU; GPU використовується, лише якщо він є і його обрано.
"""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).parent
EXAMPLES_DIR = ROOT / "test_images"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CONF_FLOOR = 0.05          # модель рахує з цим порогом, повзунок лише фільтрує результат
MAX_EXAMPLES = 24          # скільки прикладів показувати з однієї теки

INIT_ORDER = ["zeroshot", "scratch", "coco", "dota_frozen", "dota_unfrozen"]
INIT_LABELS = {
    "zeroshot": "Zero-shot (DOTA)",
    "scratch": "З нуля",
    "coco": "COCO",
    "dota_frozen": "DOTA, заморожений backbone",
    "dota_unfrozen": "DOTA, дотренування",
    "other": "Інше",
    "custom": "Власний файл .pt",
}
SIZE_ORDER = ["n", "s", "m", "l", "x"]
STAGE_RANK = {"stage2": 3, "full": 2, "frozen": 2, "stage1": 1}   # з кількох етапів лишаємо фінальний
GROUP_LABELS = {
    "vedai_test_data": "VEDAI",
    "mar20_test_data": "MAR20",
    "dior_r_test_data": "DIOR-R",
    "real_test_data": "Реальні знімки (без розмітки)",
}
PALETTE = ["#0E7C86", "#E4572E", "#E9A23B", "#5B8E3E", "#7B5EA7",
           "#C2336B", "#3D5A80", "#8D6E63", "#2BB3A3", "#A3A847", "#D1495B", "#4F6D7A"]

st.set_page_config(page_title="OBB-aerial", page_icon="🛰️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700&display=swap');
html, body, .stApp,
.stApp *:not([data-testid="stIconMaterial"]):not(.material-icons):not([class*="material-symbols"]) {
    font-family: 'Manrope', system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1280px; }
header[data-testid="stHeader"] { background: transparent; }
h1.app-title { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }
p.app-sub { color: #5B6F73; margin: .15rem 0 1.4rem; }
section[data-testid="stSidebar"] { border-right: 1px solid #D5E0DE; }
section[data-testid="stSidebar"] h3 { font-size: .95rem; font-weight: 700; margin-bottom: .2rem; }
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; }
[data-testid="stFileUploaderDropzone"] { border-radius: 14px; border: 1.5px dashed #9DB8B4; }
div[data-testid="stImage"] img { border-radius: 12px; }
.nav-title { text-align: center; line-height: 1.25; }
.nav-title b { display: block; font-size: 1rem; word-break: break-all; }
.nav-title span { color: #5B6F73; font-size: .85rem; }
.count { font-size: 2.6rem; font-weight: 700; line-height: 1; }
.count small { font-size: .95rem; font-weight: 500; color: #5B6F73; margin-left: .4rem; }
.chip { display: inline-flex; align-items: center; gap: .45rem; padding: .25rem .7rem;
        margin: 0 .35rem .4rem 0; border-radius: 999px; background: #FFFFFF;
        border: 1px solid #D5E0DE; font-size: .88rem; }
.chip i { width: .7rem; height: .7rem; border-radius: 3px; display: inline-block; }
.chip b { font-weight: 700; }
.hint { color: #5B6F73; font-size: .85rem; }
.empty { padding: 3rem 1rem; text-align: center; color: #5B6F73; border: 1.5px dashed #9DB8B4;
         border-radius: 14px; }
</style>
""",
    unsafe_allow_html=True,
)


# ───────────────────────────── моделі ─────────────────────────────
@dataclass(frozen=True)
class ModelEntry:
    path: str
    init: str
    size: str
    dataset: str = ""
    fold: str = ""
    seed: str = ""
    test_map: float | None = None      # test mAP50-95 з summary.json (якщо є)
    stage: str = ""


def resolve_dir(text: str) -> Path:
    p = Path(text).expanduser()
    return p if p.is_absolute() or not (ROOT / p).is_dir() else ROOT / p


def read_test_map(run_dir: Path, stage_name: str) -> float | None:
    """Метрики в summary.json стосуються лише фінального етапу прогону."""
    summary = run_dir / "summary.json"
    if not summary.exists():
        return None
    try:
        s = json.loads(summary.read_text())
        if list(s.get("stages", {}))[-1:] == [stage_name]:
            return float(s["final"]["test"]["mAP50-95"])
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False, ttl=30)
def discover_models(runs_dir: str) -> list[ModelEntry]:
    entries: list[ModelEntry] = []

    # офіційні DOTA-ваги: якщо файлу немає, Ultralytics докачає його сам
    for s in ("n", "s", "m"):
        name = f"yolo26{s}-obb.pt"
        entries.append(ModelEntry(str(ROOT / name) if (ROOT / name).exists() else name, "zeroshot", s))

    root = resolve_dir(runs_dir)
    if not root.is_dir():
        return entries

    # усі етапи, у кожному прогоні фінальний (найбільший STAGE_RANK) першим
    weights = sorted(root.rglob("weights/best.pt"),
                     key=lambda p: (str(p.parents[2]), -STAGE_RANK.get(p.parents[1].name, 0)))
    for p in weights:
        stage = p.parents[1].name
        run_dir = p.parents[2]
        rel = run_dir.relative_to(root).parts
        tmap = read_test_map(run_dir, stage)
        if len(rel) >= 4:
            dataset, size_dir, init, foldseed = rel[-4:]
            ms = re.search(r"yolo\d+([nsmlx])", size_dir)
            fs = re.search(r"fold0*(\d+)_seed(\d+)", foldseed)
            entries.append(ModelEntry(str(p), init, ms.group(1) if ms else size_dir, dataset,
                                      fs.group(1) if fs else foldseed, fs.group(2) if fs else "",
                                      tmap, stage=stage))
        else:
            entries.append(ModelEntry(str(p), "other", "/".join(rel), test_map=tmap, stage=stage))
    return entries


def variant_labels(variants: list[ModelEntry]) -> list[str]:
    """Підписи лише з тих полів (датасет / фолд / seed), які справді відрізняються."""
    names = {"dataset": "", "fold": "фолд ", "seed": "seed ", "stage": ""}
    differing = [f for f in names if len({getattr(v, f) for v in variants}) > 1]
    return [", ".join(f"{names[f]}{getattr(v, f)}" for f in differing) for v in variants]


def choose(label: str, options: list, key: str, fmt=str, help: str | None = None):
    """Сегментований перемикач, а на старих версіях Streamlit radio."""
    if hasattr(st, "segmented_control"):
        val = st.segmented_control(label, options, default=options[0], format_func=fmt, key=key, help=help)
        return val if val in options else options[0]
    return st.radio(label, options, format_func=fmt, key=key, horizontal=True, help=help)


@st.cache_data(show_spinner=False)
def save_upload(name: str, data: bytes) -> str:
    """Завантажений .pt -> тимчасовий файл (кешується за вмістом)."""
    tmp = Path(tempfile.mkdtemp()) / name
    tmp.write_bytes(data)
    return str(tmp)


def model_picker(runs_dir: str) -> tuple[str | None, str, float | None]:
    """Повертає (шлях до ваг, підпис, test mAP50-95)."""
    entries = discover_models(runs_dir)
    inits = [i for i in INIT_ORDER if any(e.init == i for e in entries)]
    inits += sorted({e.init for e in entries} - set(INIT_ORDER)) + ["custom"]
    default = inits.index("dota_unfrozen") if "dota_unfrozen" in inits else 0
    init = st.selectbox("Ініціалізація", inits, index=default, format_func=lambda i: INIT_LABELS.get(i, i))

    if init == "custom":
        up = st.file_uploader("Ваги (.pt)", type=["pt"])
        st.caption("Файл .pt виконує код під час завантаження: відкривайте лише власні ваги.")
        return (save_upload(up.name, up.getvalue()), up.name, None) if up else (None, "", None)

    sizes = sorted({e.size for e in entries if e.init == init},
                   key=lambda s: SIZE_ORDER.index(s) if s in SIZE_ORDER else 99)
    size = choose("Розмір моделі", sizes, key=f"size_{init}",
                  help="n: nano, s: small, m: medium. Більша модель точніша, але повільніша.")
    variants = [e for e in entries if e.init == init and e.size == size]
    entry = variants[0]
    if len(variants) > 1:
        labels = variant_labels(variants)
        entry = variants[st.selectbox("Варіант", range(len(variants)),
                                      format_func=lambda i: labels[i] or f"#{i + 1}")]
    if entry.test_map is not None:
        st.caption(f"test mAP50-95: {entry.test_map:.3f}")
    return entry.path, f"{INIT_LABELS.get(init, init)}, {size}", entry.test_map


@st.cache_resource(show_spinner="Завантаження моделі...")
def load_model(path: str, mtime: float):
    from ultralytics import YOLO
    return YOLO(path)


# ───────────────────────────── зображення ─────────────────────────────
def open_image(data: bytes) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


@st.cache_data(show_spinner=False, ttl=60)
def list_example_groups() -> dict[str, list[str]]:
    if not EXAMPLES_DIR.exists():
        return {}
    groups: dict[str, list[str]] = {}
    for d in sorted(p for p in EXAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(str(f) for f in d.rglob("*") if f.suffix.lower() in IMG_EXT)
        if files:
            groups[GROUP_LABELS.get(d.name, d.name)] = files
    loose = sorted(str(f) for f in EXAMPLES_DIR.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXT)
    if loose:
        groups["Інші"] = loose
    return groups


@st.cache_data(show_spinner=False)
def thumbnail(path: str) -> bytes:
    img = open_image(Path(path).read_bytes())
    img.thumbnail((360, 360))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=80)
    return buf.getvalue()


def example_gallery() -> list[dict]:
    groups = list_example_groups()
    if not groups:
        st.info("Теку test_images не знайдено або вона порожня.")
        return []
    # вибір зберігаємо у власному наборі: стан віджетів, яких немає на екрані, Streamlit видаляє
    sel: set[str] = st.session_state.setdefault("ex_sel", set())
    group = choose("Набір прикладів", list(groups), key="example_group")
    shown = groups[group][:MAX_EXAMPLES]

    def toggle(path: str):
        (sel.add if st.session_state[f"ex_{path}"] else sel.discard)(path)

    def set_all(value: bool):
        for f in shown:
            st.session_state[f"ex_{f}"] = value
            (sel.add if value else sel.discard)(f)

    b1, b2, b3 = st.columns([1, 1, 5], vertical_alignment="center")
    b1.button("Обрати всі", on_click=set_all, args=(True,), use_container_width=True)
    b2.button("Скинути", on_click=set_all, args=(False,), use_container_width=True)
    b3.caption(f"Обрано: {len(sel)}")

    cols = st.columns(6)
    for i, f in enumerate(shown):
        with cols[i % 6]:
            st.image(thumbnail(f), use_container_width=True)
            st.checkbox(Path(f).stem, value=f in sel, key=f"ex_{f}", on_change=toggle, args=(f,))
    if len(groups[group]) > MAX_EXAMPLES:
        st.caption(f"Показано перші {MAX_EXAMPLES} із {len(groups[group])}.")

    chosen = [f for files in groups.values() for f in files if f in sel]
    return [{"name": Path(f).name, "bytes": Path(f).read_bytes()} for f in chosen]


def collect_items() -> list[dict]:
    source = choose("Джерело знімків", ["upload", "examples"], key="source",
                    fmt={"upload": "Завантажити", "examples": "Приклади з репозиторію"}.get)
    if source == "upload":
        files = st.file_uploader("Перетягніть один або кілька знімків", type=[e[1:] for e in IMG_EXT],
                                 accept_multiple_files=True, label_visibility="collapsed")
        items = [{"name": f.name, "bytes": f.getvalue()} for f in files or []]
    else:
        items = example_gallery()
    for it in items:
        it["digest"] = hashlib.md5(it["bytes"]).hexdigest()
    return items


# ───────────────────────────── інференс ─────────────────────────────
@st.cache_data(show_spinner=False, max_entries=256)
def detect(model_path: str, mtime: float, digest: str, _data: bytes, imgsz: int, device: str) -> dict:
    """Рахуємо з мінімальним порогом, далі результат фільтрується повзунком без повторного інференсу."""
    model = load_model(model_path, mtime)
    img = open_image(_data)
    t0 = time.perf_counter()
    res = model.predict(img, imgsz=imgsz, conf=CONF_FLOOR, device=device, verbose=False)[0]
    out = {"names": dict(res.names), "seconds": time.perf_counter() - t0,
           "polys": np.zeros((0, 4, 2)), "cls": np.zeros(0, int),
           "conf": np.zeros(0), "xywhr": np.zeros((0, 5))}
    if res.obb is not None and len(res.obb):
        out.update(polys=res.obb.xyxyxyxy.cpu().numpy(), cls=res.obb.cls.cpu().numpy().astype(int),
                   conf=res.obb.conf.cpu().numpy(), xywhr=res.obb.xywhr.cpu().numpy())
    return out


def color_of(cls_id: int) -> str:
    return PALETTE[cls_id % len(PALETTE)]


def text_on(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#14262B" if 0.299 * r + 0.587 * g + 0.114 * b > 150 else "#FFFFFF"


def draw(data: bytes, det: dict, thr: float, labels: bool) -> Image.Image:
    img = open_image(data)
    d = ImageDraw.Draw(img, "RGBA")
    w = max(2, img.width // 450)
    try:
        font = ImageFont.load_default(size=max(12, img.width // 70))
    except TypeError:                       # Pillow < 10.1
        font = ImageFont.load_default()
    for poly, c, p in zip(det["polys"], det["cls"], det["conf"]):
        if p < thr:
            continue
        col = color_of(int(c))
        pts = [tuple(map(float, pt)) for pt in poly]
        d.polygon(pts, fill=col + "26")
        d.line(pts + [pts[0]], fill=col, width=w, joint="curve")
        if labels:
            tag = f"{det['names'][int(c)]} {p:.2f}"
            x, y = min(pts, key=lambda q: (q[1], q[0]))
            bb = d.textbbox((0, 0), tag, font=font)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            ty = max(0, y - th - 8)
            d.rectangle([x, ty, x + tw + 8, ty + th + 8], fill=col)
            d.text((x + 4, ty + 3), tag, fill=text_on(col), font=font)
    return img


COLS = ["клас", "впевненість", "x", "y", "довжина, px", "ширина, px", "кут, °"]


def to_df(det: dict, thr: float) -> pd.DataFrame:
    rows = [{"клас": det["names"][int(c)], "впевненість": round(float(p), 3),
             "x": round(float(cx)), "y": round(float(cy)),
             "довжина, px": round(float(max(bw, bh)), 1), "ширина, px": round(float(min(bw, bh)), 1),
             "кут, °": round(float(np.degrees(ang)), 1)}
            for (cx, cy, bw, bh, ang), c, p in zip(det["xywhr"], det["cls"], det["conf"]) if p >= thr]
    return pd.DataFrame(rows, columns=COLS).sort_values("впевненість", ascending=False, ignore_index=True)


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")      # BOM, щоб Excel бачив кирилицю


# ───────────────────────────── інтерфейс ─────────────────────────────
st.markdown('<h1 class="app-title">OBB-aerial</h1>'
            '<p class="app-sub">Орієнтовані рамки для техніки на аерознімках</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Модель")
    model_slot = st.container()
    st.markdown("### Параметри")
    thr = st.slider("Поріг впевненості", CONF_FLOOR, 0.95, 0.25, 0.05)
    labels_on = st.toggle("Підписи на рамках", value=True)
    with st.expander("Додатково"):
        runs_dir = st.text_input("Тека з прогонами", value="runs",
                                 help="Тека з прогонами з ноутбука (RUNS_DIR)")
        imgsz = st.select_slider("Розмір входу (px)", [320, 512, 640, 768, 1024, 1280], value=1024,
                                 help="Моделі треновані на 1024. Менший розмір швидший на CPU, але дрібний транспорт губиться.")
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except Exception:
            has_gpu = False
        if has_gpu:
            device = st.radio("Пристрій", ["cpu", "cuda:0"], horizontal=True)
        else:
            device = "cpu"
            st.caption("GPU не знайдено: інференс на CPU.")
    with model_slot:
        model_path, model_label, _ = model_picker(runs_dir)

items = collect_items()
if not items:
    st.markdown('<div class="empty">Завантажте знімки або виберіть їх серед прикладів.</div>',
                unsafe_allow_html=True)
    st.stop()

n = len(items)
signature = tuple(it["digest"] for it in items)
if st.session_state.get("signature") != signature:
    st.session_state["signature"] = signature
    st.session_state["pos"] = 1


def step(delta: int):
    st.session_state["pos"] = min(n, max(1, st.session_state["pos"] + delta))


if n > 1:
    c1, c2, c3 = st.columns([1, 6, 1], vertical_alignment="center")
    c1.button("‹", key="prev", on_click=step, args=(-1,), disabled=st.session_state["pos"] <= 1,
              use_container_width=True)
    c3.button("›", key="next", on_click=step, args=(1,), disabled=st.session_state["pos"] >= n,
              use_container_width=True)
    c2.markdown(f'<div class="nav-title"><b>{html.escape(items[st.session_state["pos"] - 1]["name"])}</b>'
                f'<span>{st.session_state["pos"]} з {n}</span></div>', unsafe_allow_html=True)
    st.slider("Знімок", 1, n, key="pos", label_visibility="collapsed")
pos = st.session_state["pos"] if n > 1 else 1
item = items[pos - 1]
if n == 1:
    st.markdown(f'<div class="nav-title"><b>{html.escape(item["name"])}</b></div>', unsafe_allow_html=True)

# результат зберігається, доки не зміниться набір знімків, модель чи параметри інференсу
mtime = Path(model_path).stat().st_mtime if model_path and Path(model_path).exists() else 0.0
run_key = (signature, model_path, mtime, imgsz, device)
state = st.session_state.get("run")
dets = state["dets"] if state and state["key"] == run_key else None

left, right = st.columns([3, 1.4], gap="large")

with right:
    if model_path is None:
        st.warning("Завантажте файл ваг у бічній панелі.")
    label = f"Розпізнати ({n})" if n > 1 else "Розпізнати"
    if st.button(label, type="primary", disabled=model_path is None, use_container_width=True):
        bar = st.progress(0.0, text="Розпізнавання…")
        try:
            new = []
            for i, it in enumerate(items):
                new.append(detect(model_path, mtime, it["digest"], it["bytes"], imgsz, device))
                bar.progress((i + 1) / n, text=f"Розпізнавання… {i + 1} з {n}")
        except Exception as e:
            bar.empty()
            st.error(f"Не вдалося виконати інференс: {e}")
            st.stop()
        bar.empty()
        dets = new
        st.session_state["run"] = {"key": run_key, "dets": dets}

    if dets is None:
        st.markdown('<p class="hint">Оберіть модель і натисніть «Розпізнати». '
                    'Після зміни знімків, моделі, розміру входу чи пристрою розпізнавання треба повторити.</p>',
                    unsafe_allow_html=True)

if dets is None:
    left.image(open_image(item["bytes"]), use_container_width=True)
    st.stop()

det = dets[pos - 1]
df = to_df(det, thr)
left.image(draw(item["bytes"], det, thr, labels_on), use_container_width=True)

with right:
    st.markdown(f'<div class="count">{len(df)}<small>об\'єктів</small></div>', unsafe_allow_html=True)
    st.markdown(f'<p class="hint">{html.escape(model_label)}, поріг {thr:.2f}, {det["seconds"]:.2f} с, '
                f'{device}</p>', unsafe_allow_html=True)
    if len(df):
        inv = {v: k for k, v in det["names"].items()}
        st.markdown("".join(
            f'<span class="chip"><i style="background:{color_of(inv[name])}"></i>{html.escape(name)} <b>{cnt}</b></span>'
            for name, cnt in df["клас"].value_counts().items()), unsafe_allow_html=True)
    else:
        st.markdown('<p class="hint">Нічого не знайдено. Спробуйте нижчий поріг або більший розмір входу.</p>',
                    unsafe_allow_html=True)
    st.download_button("CSV цього знімка", csv_bytes(df), f"{Path(item['name']).stem}_detections.csv",
                       "text/csv", use_container_width=True, disabled=df.empty)
    if n > 1:
        all_df = pd.concat([to_df(d, thr).assign(файл=it["name"]) for it, d in zip(items, dets)],
                           ignore_index=True)
        st.download_button(f"CSV усіх знімків ({n})", csv_bytes(all_df), "detections.csv", "text/csv",
                           use_container_width=True, disabled=all_df.empty)

if len(df):
    with st.expander("Усі детекції"):
        st.dataframe(df, hide_index=True, use_container_width=True)
