"""Демо: розпізнавання транспорту на аерознімках (YOLO26-OBB, VEDAI).

Запуск:  streamlit run app.py
Моделі:  усі best.pt з теки прогонів (структура з ноутбука: runs/vedai/yolo26n/<init>/foldNN_seedS/<етап>/weights/best.pt),
         офіційні DOTA-ваги (zero-shot) або власний .pt файл.
Працює на CPU; GPU використовується, лише якщо він є і його обрано.
"""
import json
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

PRETRAINED = {f"DOTA zero-shot: yolo26{s}-obb.pt": f"yolo26{s}-obb.pt" for s in ("n", "s", "m")}
STAGE_ORDER = {"stage2": 0, "full": 1, "frozen": 2, "stage1": 3}     # фінальний етап прогону першим

st.set_page_config(page_title="Розпізнавання транспорту (YOLO26-OBB)", layout="wide")


def find_models(runs_dir):
    """{підпис: шлях до best.pt}. Якщо поруч є summary.json, у підписі показується test mAP50-95."""
    root, found = Path(runs_dir).expanduser(), {}
    if not root.is_dir():
        return found
    weights = sorted(root.rglob("weights/best.pt"),
                     key=lambda p: (str(p.parents[2]), STAGE_ORDER.get(p.parents[1].name, 9)))
    for w in weights:
        stage_dir, run_dir = w.parents[1], w.parents[2]
        label = f"{run_dir.relative_to(root)} · {stage_dir.name}"
        summary = run_dir / "summary.json"
        if summary.exists():
            try:
                s = json.loads(summary.read_text())
                last_stage = list(s.get("stages", {}))[-1:]
                if last_stage == [stage_dir.name]:            # метрики в summary стосуються лише фінального етапу
                    label += f" · test mAP50-95 {s['final']['test']['mAP50-95']:.3f}"
            except Exception:
                pass
        found[label] = str(w)
    return found


@st.cache_resource(show_spinner="Завантаження моделі...")
def load_model(path):
    return YOLO(path)


@st.cache_data(show_spinner=False)
def save_upload(name, data):
    """Завантажений .pt -> тимчасовий файл (шлях кешується за вмістом)."""
    tmp = Path(tempfile.mkdtemp()) / name
    tmp.write_bytes(data)
    return str(tmp)


# ---------------- бічна панель ----------------
with st.sidebar:
    st.header("Модель")
    runs_dir = st.text_input("Тека з прогонами", value="runs", help="Тека, у якій лежать прогони з ноутбука (RUNS_DIR)")
    models = {**find_models(runs_dir), **PRETRAINED}
    upload_label = "Власний файл .pt"
    choice = st.selectbox("Модель", list(models) + [upload_label])
    model_path = models.get(choice)
    if choice == upload_label:
        up = st.file_uploader("Ваги (.pt)", type=["pt"])
        st.caption("Файл .pt виконує код під час завантаження: відкривайте лише власні ваги.")
        model_path = save_upload(up.name, up.getvalue()) if up else None
    if not find_models(runs_dir):
        st.caption("У цій теці немає best.pt: доступні лише DOTA-ваги та власний файл.")

    st.header("Параметри")
    conf = st.slider("Поріг впевненості", 0.05, 0.95, 0.25, 0.05)
    imgsz = st.select_slider("Розмір входу (px)", options=[320, 512, 640, 768, 1024, 1280], value=1024,
                             help="Моделі треновані на 1024. Менший розмір швидший на CPU, але дрібний транспорт губиться.")
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    device = st.radio("Пристрій", devices, horizontal=True) if len(devices) > 1 else "cpu"
    if len(devices) == 1:
        st.caption("GPU не знайдено: інференс на CPU.")

# ---------------- основна частина ----------------
st.title("Розпізнавання транспорту на аерознімках")
st.caption("YOLO26-OBB, дотренована на VEDAI: орієнтовані рамки для car, truck, pickup, van, camping_car, tractor, boat, plane, other.")

file = st.file_uploader("Знімок", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"])
if file is None:
    st.info("Завантажте аерознімок, оберіть модель у бічній панелі та натисніть «Розпізнати».")
    st.stop()

image = Image.open(file).convert("RGB")
run = st.button("Розпізнати", type="primary", disabled=model_path is None)
if model_path is None:
    st.warning("Оберіть модель або завантажте файл ваг.")

# результат зберігається між перемальовуваннями сторінки, доки не зміниться знімок, модель чи параметри
key = (file.name, file.size, model_path, conf, imgsz, device)
if run:
    try:
        model = load_model(model_path)
        t0 = time.perf_counter()
        res = model.predict(image, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]
        seconds = time.perf_counter() - t0
    except Exception as e:
        st.error(f"Не вдалося виконати інференс: {e}")
        st.stop()
    obb = res.obb
    rows = []
    if obb is not None and len(obb):
        xywhr = obb.xywhr.cpu().numpy()
        for (x, y, w, h, r), c, p in zip(xywhr, obb.cls.cpu().numpy(), obb.conf.cpu().numpy()):
            rows.append({"клас": res.names[int(c)], "впевненість": round(float(p), 3), "x": round(float(x)),
                         "y": round(float(y)), "довжина, px": round(float(max(w, h)), 1),
                         "ширина, px": round(float(min(w, h)), 1), "кут, °": round(float(r) * 57.2958, 1)})
    st.session_state["result"] = {
        "key": key, "seconds": seconds, "rows": rows,
        "plot": res.plot(line_width=2, font_size=12)[..., ::-1],      # BGR -> RGB
    }

result = st.session_state.get("result")
left, right = st.columns(2)
left.subheader("Знімок")
left.image(image, use_container_width=True)
right.subheader("Результат")
if not result or result["key"] != key:
    right.info("Натисніть «Розпізнати».")
    st.stop()

right.image(result["plot"], use_container_width=True)
df = pd.DataFrame(result["rows"])
m1, m2, m3 = st.columns(3)
m1.metric("Знайдено об'єктів", len(df))
m2.metric("Час інференсу", f"{result['seconds']:.2f} с")
m3.metric("Пристрій", device)
if df.empty:
    st.warning("Нічого не знайдено. Спробуйте нижчий поріг впевненості або більший розмір входу.")
    st.stop()

c1, c2 = st.columns([1, 2])
counts = df["клас"].value_counts().rename_axis("клас").reset_index(name="кількість")
c1.subheader("По класах")
c1.dataframe(counts, hide_index=True, use_container_width=True)
c2.subheader("Усі детекції")
c2.dataframe(df.sort_values("впевненість", ascending=False), hide_index=True, use_container_width=True)
st.download_button("Завантажити детекції (CSV)", df.to_csv(index=False).encode("utf-8"),
                   file_name=f"{Path(file.name).stem}_detections.csv", mime="text/csv")
