"""
Parallel live visual demo.

Left panel  — Recording Studio: live camera, hand skeleton, training controls,
              device physics sliders, conductance heatmap.
Right panel — System Comparison: all 4 systems receive the same input
              simultaneously; confidence bars, accuracy plot, latency table.

Usage:
    cd C:/Users/rober/Desktop/Experiment_Mem
    .venv/Scripts/python scripts/demo_visual.py
    .venv/Scripts/python scripts/demo_visual.py --class-names A B C D E
"""

import sys, os, argparse, threading, time
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dearpygui.dearpygui as dpg

from memristor.device_model import MemristorDeviceModel
from network.classifier import MemristorClassifier
from learning.continual_trainer import ContinualTrainer
from baselines.frozen_linear import FrozenLinearBaseline, FrozenLinearTrainer
from baselines.mlp_sgd import MLPOnlineBaseline, MLPOnlineTrainer
from baselines.cnn_online import CNNOnlineBaseline, CNNOnlineTrainer
from pipeline.capture import CameraCapture
from pipeline.landmark_extractor import LandmarkExtractor

# ── layout ────────────────────────────────────────────────────────
WIN_W, WIN_H = 1560, 900
LEFT_W       = 530
RIGHT_W      = WIN_W - LEFT_W - 12
CAM_W, CAM_H = 490, 340
CMAP_W, CMAP_H = 490, 90
HIST_LEN    = 300
MAX_CLASSES = 10
CONF_BAR_W  = 180

SYSTEMS    = ["memristor", "frozen_lin", "mlp_sgd", "cnn_online"]
SYS_LABELS = {
    "memristor":  "Memristor (ours)",
    "frozen_lin": "Frozen + Linear",
    "mlp_sgd":    "MLP + SGD",
    "cnn_online": "CNN Online",
}
SYS_COLORS = {
    "memristor":  (80,  200, 120),
    "frozen_lin": (100, 160, 240),
    "mlp_sgd":    (240, 160,  80),
    "cnn_online": (220, 100, 100),
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ══════════════════════════════════════════════════════════════════
#  Shared state
# ══════════════════════════════════════════════════════════════════

class State:
    def __init__(self, class_names: list[str]):
        self.lock        = threading.Lock()
        self.class_names = list(class_names)
        self.n_classes   = len(class_names)

        self.frame_rgba  = np.zeros((CAM_H, CAM_W, 4), dtype=np.float32)
        self.hand_detected = False

        self.probs   = {s: np.zeros(len(class_names), dtype=np.float32) for s in SYSTEMS}
        self.pred    = {s: 0 for s in SYSTEMS}
        self.latency = {s: (0.0, 0.0) for s in SYSTEMS}

        self.acc_hist  = {s: deque(maxlen=HIST_LEN) for s in SYSTEMS}
        self.sample_idx = 0

        self.cond_rgba = np.zeros((CMAP_H, CMAP_W, 4), dtype=np.float32)

        self.training_label: int | None = None
        self.n_collected    = 0
        self.target_samples = 80
        self.fps            = 0.0

    def add_class(self, name: str) -> int:
        with self.lock:
            idx = len(self.class_names)
            self.class_names.append(name)
            self.n_classes += 1
            for s in SYSTEMS:
                old = self.probs[s]
                new = np.zeros(idx + 1, dtype=np.float32)
                new[:len(old)] = old
                self.probs[s] = new
        return idx


# ══════════════════════════════════════════════════════════════════
#  Inference / training thread
# ══════════════════════════════════════════════════════════════════

class InferenceThread(threading.Thread):
    def __init__(self, models, trainers, state: State, device: torch.device):
        super().__init__(daemon=True)
        self.models   = models
        self.trainers = trainers
        self.state    = state
        self.device   = device
        self._running = True
        self._cam     = CameraCapture()
        self._ext     = LandmarkExtractor()
        self._fps_t   = time.monotonic()
        self._fps_n   = 0

    def run(self):
        self._cam.start()
        while self._running:
            frame = self._cam.get_frame(timeout=0.05)
            if frame is None:
                continue

            feats = self._ext.extract(frame)
            hand  = feats is not None

            probs_all = {s: np.zeros(self.state.n_classes, dtype=np.float32) for s in SYSTEMS}
            pred_all  = {s: 0 for s in SYSTEMS}
            lat_all   = {s: (0.0, 0.0) for s in SYSTEMS}
            acc_entry = {}
            label     = self.state.training_label

            if hand:
                x = feats.unsqueeze(0).to(self.device)
                for sys in SYSTEMS:
                    model   = self.models[sys]
                    trainer = self.trainers[sys]

                    t0 = time.perf_counter()
                    model.eval()
                    with torch.no_grad():
                        logits = model(x)
                    inf_ms = (time.perf_counter() - t0) * 1000

                    p    = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    full = np.zeros(self.state.n_classes, dtype=np.float32)
                    full[:len(p)] = p
                    probs_all[sys] = full
                    pred_all[sys]  = int(np.argmax(full))

                    upd_ms = 0.0
                    if label is not None:
                        t0 = time.perf_counter()
                        result = trainer.step(feats.unsqueeze(0), torch.tensor([label]))
                        upd_ms = (time.perf_counter() - t0) * 1000
                        acc_entry[sys] = result.get("acc", 0.0)

                    lat_all[sys] = (inf_ms, upd_ms)

                if label is not None:
                    with self.state.lock:
                        self.state.n_collected += 1
                        self.state.sample_idx  += 1
                        si = self.state.sample_idx
                    for sys in SYSTEMS:
                        if sys in acc_entry:
                            self.state.acc_hist[sys].append((si, acc_entry[sys]))

            # FPS
            self._fps_n += 1
            elapsed = time.monotonic() - self._fps_t
            if elapsed >= 1.0:
                fps = self._fps_n / elapsed
                self._fps_n = 0
                self._fps_t = time.monotonic()
            else:
                fps = self.state.fps

            # Annotate frame and convert to float32 RGBA for DPG
            annotated = _draw_skeleton(frame, feats, fps, collecting=(label is not None and hand))
            resized   = cv2.resize(annotated, (CAM_W, CAM_H))
            rgba_u8   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)
            rgba_f32  = rgba_u8.astype(np.float32) / 255.0

            # Conductance heatmap
            cmap_f32 = _conductance_heatmap(self.models["memristor"], CMAP_W, CMAP_H)

            with self.state.lock:
                self.state.frame_rgba    = rgba_f32
                self.state.hand_detected = hand
                self.state.probs         = probs_all
                self.state.pred          = pred_all
                self.state.latency       = lat_all
                self.state.cond_rgba     = cmap_f32
                self.state.fps           = fps

    def stop(self):
        self._running = False
        self._cam.stop()
        self._ext.close()


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

def _draw_skeleton(frame: np.ndarray, feats, fps: float, collecting: bool = False) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if feats is not None:
        pts = feats.reshape(21, 3).numpy()
        p2  = pts[:, :2].copy()
        p2 -= p2.min(0)
        r   = p2.max()
        if r > 0:
            p2 /= r
        p2 = (p2 * 0.72 + 0.14) * np.array([w, h])
        p2 = p2.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(out, tuple(p2[a]), tuple(p2[b]), (60, 220, 60), 2)
        for i, pt in enumerate(p2):
            cv2.circle(out, tuple(pt), 5,
                       (0, 255, 180) if i == 0 else (0, 190, 255), -1)

        # Green "HAND DETECTED" banner
        cv2.rectangle(out, (0, h - 36), (w, h), (0, 180, 0), -1)
        cv2.putText(out, "HAND DETECTED", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if collecting:
            cv2.putText(out, "● COLLECTING", (w - 180, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    else:
        # Red "SHOW HAND" banner
        cv2.rectangle(out, (0, h - 36), (w, h), (0, 0, 180), -1)
        cv2.putText(out, "SHOW HAND TO CAMERA", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(out, f"{fps:.0f} FPS", (w - 85, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return out


def _conductance_heatmap(model, w: int, h: int) -> np.ndarray:
    layer = model.output_layer
    g_eff = (layer.G_pos - layer.G_neg).detach().cpu().float().numpy()
    mn, mx = g_eff.min(), g_eff.max()
    span  = mx - mn if mx - mn > 1e-12 else 1.0
    g8    = ((g_eff - mn) / span * 255).astype(np.uint8)
    rsz   = cv2.resize(g8, (w, h), interpolation=cv2.INTER_NEAREST)
    col   = cv2.applyColorMap(rsz, cv2.COLORMAP_PLASMA)
    rgba  = cv2.cvtColor(col, cv2.COLOR_BGR2RGBA)
    return rgba.astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════
#  Dear PyGui UI
# ══════════════════════════════════════════════════════════════════

class VisualDemo:
    def __init__(self, state: State, inf_thread: InferenceThread,
                 models: dict, trainers: dict):
        self.state      = state
        self.inf_thread = inf_thread
        self.models     = models
        self.trainers   = trainers

    # ── build ──────────────────────────────────────────────────────

    def build(self):
        dpg.create_context()
        dpg.create_viewport(
            title="Memristor Hand-Sign Demo",
            width=WIN_W, height=WIN_H, resizable=False,
        )
        dpg.setup_dearpygui()

        # Textures — use flat numpy float32 arrays (no .tolist())
        with dpg.texture_registry():
            dpg.add_raw_texture(
                CAM_W, CAM_H,
                default_value=np.zeros(CAM_H * CAM_W * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba, tag="tex_cam",
            )
            dpg.add_raw_texture(
                CMAP_W, CMAP_H,
                default_value=np.zeros(CMAP_H * CMAP_W * 4, dtype=np.float32),
                format=dpg.mvFormat_Float_rgba, tag="tex_cmap",
            )

        # Per-system bar themes
        for sys in SYSTEMS:
            r, g, b = SYS_COLORS[sys]
            with dpg.theme(tag=f"theme_{sys}"):
                with dpg.theme_component(dpg.mvProgressBar):
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram, (r, g, b, 210))

        with dpg.window(tag="root", no_title_bar=True, no_move=True,
                        no_resize=True, width=WIN_W, height=WIN_H):
            with dpg.group(horizontal=True):
                self._build_left()
                self._build_right()

        dpg.set_primary_window("root", True)
        dpg.show_viewport()

    # ── left panel ─────────────────────────────────────────────────

    def _build_left(self):
        with dpg.child_window(width=LEFT_W, height=WIN_H - 16, border=True):
            dpg.add_text("◉  RECORDING STUDIO", color=(255, 220, 80))
            dpg.add_separator()

            dpg.add_image("tex_cam", width=CAM_W, height=CAM_H)
            dpg.add_text("Waiting for camera…", tag="txt_status",
                         color=(160, 160, 160))

            dpg.add_separator()
            dpg.add_text("Train a new sign:", color=(200, 200, 160))
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint="Sign name…", tag="inp_name", width=150)
                dpg.add_button(label="▶  Start", tag="btn_start",
                               callback=self._cb_start)
                dpg.add_button(label="■  Stop", tag="btn_stop",
                               callback=self._cb_stop, show=False)
            dpg.add_progress_bar(default_value=0.0, tag="bar_prog",
                                 width=LEFT_W - 24, height=18, overlay="0 / 80")

            dpg.add_separator()
            dpg.add_text("Device physics (live):", color=(200, 200, 160))
            dpg.add_slider_float(label="Read noise σ",      tag="sl_read",
                                 default_value=0.01, min_value=0.0, max_value=0.25,
                                 width=280, callback=self._cb_device)
            dpg.add_slider_float(label="Write noise σ",     tag="sl_write",
                                 default_value=0.05, min_value=0.0, max_value=0.35,
                                 width=280, callback=self._cb_device)
            dpg.add_slider_int(label="Conductance levels",  tag="sl_levels",
                               default_value=32, min_value=2, max_value=256,
                               width=280, callback=self._cb_device)
            dpg.add_checkbox(label="Read noise during inference",
                             tag="chk_noise", default_value=True,
                             callback=self._cb_toggle_noise)

            dpg.add_separator()
            dpg.add_text("Output layer  G⁺ − G⁻", color=(200, 200, 160))
            dpg.add_image("tex_cmap", width=CMAP_W, height=CMAP_H)
            with dpg.group(horizontal=True):
                dpg.add_text("low", color=(80, 80, 220))
                dpg.add_text("                                      high",
                             color=(220, 80, 80))

    # ── right panel ────────────────────────────────────────────────

    def _build_right(self):
        with dpg.child_window(width=RIGHT_W, height=WIN_H - 16, border=False):
            dpg.add_text("⬡  SYSTEM COMPARISON  (same input, same labels)",
                         color=(255, 220, 80))
            dpg.add_separator()

            # Row 1: memristor + frozen_lin
            with dpg.group(horizontal=True):
                for sys in SYSTEMS[:2]:
                    self._build_system_panel(sys)

            # Row 2: mlp_sgd + cnn_online
            with dpg.group(horizontal=True):
                for sys in SYSTEMS[2:]:
                    self._build_system_panel(sys)

            dpg.add_separator()

            # Accuracy plot
            with dpg.child_window(width=RIGHT_W - 8, height=205, border=True):
                dpg.add_text("Accuracy over time  (all 4 systems)",
                             color=(200, 200, 160))
                with dpg.plot(height=178, width=RIGHT_W - 22,
                              no_title=True, tag="acc_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Samples", tag="ax_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Acc",     tag="ax_y")
                    dpg.set_axis_limits("ax_y", 0.0, 1.0)
                    for sys in SYSTEMS:
                        r, g, b = SYS_COLORS[sys]
                        dpg.add_line_series([], [], label=SYS_LABELS[sys],
                                            parent="ax_y", tag=f"acc_{sys}")

            # Latency table
            with dpg.child_window(width=RIGHT_W - 8, height=95, border=True):
                dpg.add_text("Latency", color=(200, 200, 160))
                with dpg.table(header_row=True, borders_innerH=True,
                               borders_outerH=True):
                    dpg.add_table_column(label="System",    width_fixed=True, init_width_or_weight=155)
                    dpg.add_table_column(label="Inference", width_fixed=True, init_width_or_weight=95)
                    dpg.add_table_column(label="Update",    width_fixed=True, init_width_or_weight=95)
                    dpg.add_table_column(label="GPU MB",    width_fixed=True, init_width_or_weight=80)
                    for sys in SYSTEMS:
                        with dpg.table_row():
                            dpg.add_text(SYS_LABELS[sys], color=SYS_COLORS[sys])
                            dpg.add_text("—", tag=f"lat_inf_{sys}")
                            dpg.add_text("—", tag=f"lat_upd_{sys}")
                            dpg.add_text("—", tag=f"lat_mem_{sys}")

    def _build_system_panel(self, sys: str):
        color = SYS_COLORS[sys]
        panel_w = (RIGHT_W // 2) - 8
        with dpg.child_window(width=panel_w, height=265, border=True):
            dpg.add_text(SYS_LABELS[sys], color=color)
            dpg.add_separator()
            dpg.add_text("No hand", tag=f"pred_{sys}", color=(160, 160, 160))
            dpg.add_separator()
            for i in range(MAX_CLASSES):
                show = i < self.state.n_classes
                nm   = self.state.class_names[i] if i < len(self.state.class_names) else f"cls{i}"
                with dpg.group(horizontal=True, tag=f"crow_{sys}_{i}", show=show):
                    dpg.add_text(f"{nm:<9}", tag=f"clbl_{sys}_{i}")
                    bar = dpg.add_progress_bar(
                        default_value=0.0, tag=f"cbar_{sys}_{i}",
                        width=CONF_BAR_W, height=18, overlay="0%",
                    )
                    dpg.bind_item_theme(bar, f"theme_{sys}")

    # ── callbacks ──────────────────────────────────────────────────

    def _cb_start(self, *_):
        name = dpg.get_value("inp_name").strip()
        if not name:
            return

        # Reuse existing class if name already known, otherwise add new
        names = self.state.class_names
        if name in names:
            idx = names.index(name)
        else:
            idx = self.state.add_class(name)
            # Expand all 4 models to accommodate the new output neuron
            for sys, model in self.models.items():
                while idx >= model.n_classes:
                    model.add_class()
            # Reinitialise optimisers so new parameters are included
            import torch.optim as optim
            for sys, trainer in self.trainers.items():
                if hasattr(trainer, "optimizer"):
                    trainer.optimizer = optim.Adam(
                        [p for p in self.models[sys].parameters()
                         if p.requires_grad],
                        lr=0.001,
                    )
            # Reveal new row in all 4 system panels
            for sys in SYSTEMS:
                tag = f"crow_{sys}_{idx}"
                if dpg.does_item_exist(tag):
                    dpg.configure_item(tag, show=True)
                    dpg.configure_item(f"clbl_{sys}_{idx}",
                                       default_value=f"{name:<9}")

        self.state.training_label = idx
        self.state.n_collected    = 0
        dpg.hide_item("btn_start")
        dpg.show_item("btn_stop")

    def _cb_stop(self, *_):
        self.state.training_label = None
        dpg.show_item("btn_start")
        dpg.hide_item("btn_stop")

    def _cb_device(self, *_):
        dm = self.models["memristor"].dm
        dm.sigma_read  = dpg.get_value("sl_read")
        dm.sigma_write = dpg.get_value("sl_write")
        dm.n_levels    = dpg.get_value("sl_levels")

    def _cb_toggle_noise(self, *_):
        on = dpg.get_value("chk_noise")
        for layer in self.models["memristor"].memristor_layers():
            layer.apply_noise = on

    # ── per-frame update ───────────────────────────────────────────

    def update(self):
        with self.state.lock:
            frame_f   = self.state.frame_rgba         # already float32
            cmap_f    = self.state.cond_rgba
            hand      = self.state.hand_detected
            probs_all = {s: self.state.probs[s].copy() for s in SYSTEMS}
            pred_all  = dict(self.state.pred)
            lat_all   = dict(self.state.latency)
            n_col     = self.state.n_collected
            t_lbl     = self.state.training_label
            names     = list(self.state.class_names)
            acc_hists = {s: list(self.state.acc_hist[s]) for s in SYSTEMS}

        # Textures — pass numpy arrays directly, no .tolist()
        dpg.set_value("tex_cam",  frame_f.flatten())
        dpg.set_value("tex_cmap", cmap_f.flatten())

        # Status
        if hand:
            p0   = pred_all["memristor"]
            nm   = names[p0] if p0 < len(names) else f"cls{p0}"
            conf = float(probs_all["memristor"][p0])
            dpg.configure_item("txt_status",
                               default_value=f"✓  {nm}  ({conf:.0%})",
                               color=(80, 255, 140))
        else:
            dpg.configure_item("txt_status",
                               default_value="No hand detected",
                               color=(160, 160, 160))

        # Training progress
        if t_lbl is not None:
            ratio = min(n_col / max(self.state.target_samples, 1), 1.0)
            dpg.set_value("bar_prog", ratio)
            dpg.configure_item("bar_prog",
                               overlay=f"{n_col} / {self.state.target_samples}")
            if n_col >= self.state.target_samples:
                self._cb_stop()

        # Confidence bars — all 4 systems, same tag pattern
        for sys in SYSTEMS:
            probs = probs_all[sys]
            pred  = pred_all[sys]
            nm    = names[pred] if pred < len(names) else f"cls{pred}"
            c     = float(probs[pred]) if pred < len(probs) else 0.0
            dpg.configure_item(f"pred_{sys}",
                               default_value=f"{nm}  {c:.0%}" if hand else "No hand")
            for i in range(min(len(names), MAX_CLASSES)):
                v = float(probs[i]) if i < len(probs) else 0.0
                dpg.configure_item(f"cbar_{sys}_{i}",
                                   default_value=v, overlay=f"{v:.0%}")

        # Accuracy plot
        for sys in SYSTEMS:
            hist = acc_hists[sys]
            if hist:
                dpg.set_value(f"acc_{sys}",
                              [[h[0] for h in hist], [h[1] for h in hist]])
        dpg.fit_axis_data("ax_x")

        # Latency table
        mem_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        for sys in SYSTEMS:
            inf_ms, upd_ms = lat_all[sys]
            dpg.configure_item(f"lat_inf_{sys}", default_value=f"{inf_ms:.1f} ms")
            dpg.configure_item(f"lat_upd_{sys}", default_value=f"{upd_ms:.1f} ms")
            dpg.configure_item(f"lat_mem_{sys}", default_value=f"{mem_mb:.0f}")

    def run(self):
        self.inf_thread.start()
        while dpg.is_dearpygui_running():
            self.update()
            dpg.render_dearpygui_frame()
        self.inf_thread.stop()
        dpg.destroy_context()


# ══════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--class-names",    nargs="+", default=["A", "B", "C", "D", "E"])
    p.add_argument("--model-path",     default=None)
    p.add_argument("--target-samples", type=int, default=80)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("config/network.yaml")   as f: net_cfg = yaml.safe_load(f)
    with open("config/memristor.yaml") as f: mem_cfg = yaml.safe_load(f)

    n_cls = max(net_cfg["initial_classes"], len(args.class_names))
    dm    = MemristorDeviceModel(mem_cfg)

    mem_model = MemristorClassifier(n_cls, net_cfg["input_dim"],
                                    net_cfg["hidden_dims"], dm).to(device)
    frz_model = FrozenLinearBaseline(n_cls).to(device)
    mlp_model = MLPOnlineBaseline(n_cls).to(device)
    cnn_model = CNNOnlineBaseline(n_cls).to(device)

    if args.model_path and Path(args.model_path).exists():
        mem_model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded: {args.model_path}")

    models = {
        "memristor":  mem_model,
        "frozen_lin": frz_model,
        "mlp_sgd":    mlp_model,
        "cnn_online": cnn_model,
    }
    trainers = {
        "memristor":  ContinualTrainer(mem_model, device),
        "frozen_lin": FrozenLinearTrainer(frz_model, device),
        "mlp_sgd":    MLPOnlineTrainer(mlp_model, device),
        "cnn_online": CNNOnlineTrainer(cnn_model, device),
    }

    state  = State(args.class_names[:n_cls])
    state.target_samples = args.target_samples

    thread = InferenceThread(models, trainers, state, device)
    demo   = VisualDemo(state, thread, models, trainers)
    demo.build()
    demo.run()


if __name__ == "__main__":
    main()
