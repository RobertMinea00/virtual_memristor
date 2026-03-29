"""
Parallel live visual demo.

Layout (single wide DPG window, two clearly separated halves):

  ┌──────────────────────────────────┬──────────────────────────────────────────┐
  │  RECORDING STUDIO                │  SYSTEM COMPARISON  (same input)         │
  │                                  │                                          │
  │  ┌─────────────────────────┐     │  ┌─────────────┐  ┌─────────────┐       │
  │  │  Live camera + skeleton │     │  │  Memristor  │  │ Frozen+Lin  │       │
  │  │                         │     │  │  conf bars  │  │  conf bars  │       │
  │  └─────────────────────────┘     │  └─────────────┘  └─────────────┘       │
  │                                  │  ┌─────────────┐  ┌─────────────┐       │
  │  [Sign name ___] [▶ Train] [■]   │  │   MLP SGD   │  │  CNN Online │       │
  │  Samples ████████░░  64/80       │  │  conf bars  │  │  conf bars  │       │
  │                                  │  └─────────────┘  └─────────────┘       │
  │  Device params (sliders)         │  ┌──────────────────────────────────┐   │
  │  Read noise  ───●──              │  │  Accuracy over time (all 4 sys.) │   │
  │  Write noise ──●───              │  │                                  │   │
  │  #Levels     ──────●             │  └──────────────────────────────────┘   │
  │  [noise checkbox]                │                                          │
  │                                  │  Memristor update: 37 ms   Inf: 2.4 ms  │
  │  Output layer conductance        │  FrozenLin update: 1.5 ms  Inf: 0.3 ms  │
  │  [plasma heatmap]                │  MLP-SGD  update: 1.7 ms   Inf: 0.3 ms  │
  │                                  │  CNN      update: 3.4 ms   Inf: 0.4 ms  │
  └──────────────────────────────────┴──────────────────────────────────────────┘

Usage:
    cd C:/Users/rober/Desktop/Experiment_Mem
    .venv/Scripts/python scripts/demo_visual.py
    .venv/Scripts/python scripts/demo_visual.py --class-names A B C D E --model-path checkpoints/model.pt
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
WIN_W       = 1560
WIN_H       = 900
LEFT_W      = 530          # recording studio
RIGHT_W     = WIN_W - LEFT_W - 12
CAM_W, CAM_H = 490, 368
CMAP_W, CMAP_H = 490, 100
HIST_LEN    = 300
MAX_CLASSES = 10
CONF_BAR_W  = 195

SYSTEMS = ["memristor", "frozen_lin", "mlp_sgd", "cnn_online"]
SYS_LABELS = {
    "memristor":  "Memristor (ours)",
    "frozen_lin": "Frozen + Linear",
    "mlp_sgd":    "MLP + SGD",
    "cnn_online": "CNN Online",
}
SYS_COLORS = {
    "memristor":  (80, 200, 120),
    "frozen_lin": (100, 160, 240),
    "mlp_sgd":    (240, 160, 80),
    "cnn_online": (220, 100, 100),
}


# ══════════════════════════════════════════════════════════════════
#  Shared state
# ══════════════════════════════════════════════════════════════════

class State:
    def __init__(self, class_names):
        self.lock = threading.Lock()
        self.class_names = list(class_names)
        self.n_classes = len(class_names)

        # Camera
        self.frame_rgba = np.zeros((CAM_H, CAM_W, 4), dtype=np.uint8)
        self.hand_detected = False

        # Per-system predictions  {sys: np.array shape (n_classes,)}
        self.probs = {s: np.zeros(len(class_names)) for s in SYSTEMS}
        self.pred  = {s: 0 for s in SYSTEMS}

        # Per-system latency stats  {sys: (inf_ms, upd_ms)}
        self.latency = {s: (0.0, 0.0) for s in SYSTEMS}

        # Accuracy history  {sys: deque of (sample_idx, acc)}
        self.acc_hist = {s: deque(maxlen=HIST_LEN) for s in SYSTEMS}
        self.sample_idx = 0

        # Conductance heatmap (memristor output layer)
        self.cond_rgba = np.zeros((CMAP_H, CMAP_W, 4), dtype=np.uint8)

        # Training
        self.training_label: int | None = None
        self.n_collected = 0
        self.target_samples = 80
        self.fps = 0.0

    def add_class(self, name: str) -> int:
        with self.lock:
            idx = len(self.class_names)
            self.class_names.append(name)
            self.n_classes += 1
            for s in SYSTEMS:
                old = self.probs[s]
                new = np.zeros(idx + 1)
                new[:idx] = old
                self.probs[s] = new
        return idx


# ══════════════════════════════════════════════════════════════════
#  Inference thread  (feeds all 4 systems the same features)
# ══════════════════════════════════════════════════════════════════

class InferenceThread(threading.Thread):
    def __init__(self, models: dict, trainers: dict, state: State,
                 device: torch.device):
        super().__init__(daemon=True)
        self.models   = models    # {sys_name: model}
        self.trainers = trainers  # {sys_name: trainer}
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

            probs_all = {s: np.zeros(self.state.n_classes) for s in SYSTEMS}
            pred_all  = {s: 0 for s in SYSTEMS}
            lat_all   = {s: (0.0, 0.0) for s in SYSTEMS}
            acc_entry  = {}
            label      = self.state.training_label

            if hand:
                x = feats.unsqueeze(0).to(self.device)

                for sys in SYSTEMS:
                    model   = self.models[sys]
                    trainer = self.trainers[sys]
                    n       = model.n_classes

                    # ── inference latency ──
                    if self.device.type == "cuda":
                        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        e0.record()
                        model.eval()
                        with torch.no_grad():
                            logits = model(x)
                        e1.record(); torch.cuda.synchronize()
                        inf_ms = e0.elapsed_time(e1)
                    else:
                        t0 = time.perf_counter()
                        model.eval()
                        with torch.no_grad():
                            logits = model(x)
                        inf_ms = (time.perf_counter() - t0) * 1000

                    p = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    full = np.zeros(self.state.n_classes)
                    full[:n] = p
                    probs_all[sys] = full
                    pred_all[sys]  = int(np.argmax(full))

                    # ── online training (same label for all systems) ──
                    if label is not None:
                        t0 = time.perf_counter()
                        result = trainer.step(
                            feats.unsqueeze(0), torch.tensor([label])
                        )
                        upd_ms = (time.perf_counter() - t0) * 1000
                        acc_entry[sys] = result.get("acc", 0.0)
                    else:
                        upd_ms = 0.0

                    lat_all[sys] = (inf_ms, upd_ms)

                # Sample counter + collection
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
            if (elapsed := time.monotonic() - self._fps_t) >= 1.0:
                fps = self._fps_n / elapsed
                self._fps_n = 0; self._fps_t = time.monotonic()
            else:
                fps = self.state.fps

            # Conductance heatmap
            mem_model = self.models["memristor"]
            cmap = _conductance_heatmap(mem_model, CMAP_W, CMAP_H)

            # Draw skeleton on frame
            annotated = _draw_skeleton(frame, feats, fps)
            resized   = cv2.resize(annotated, (CAM_W, CAM_H))
            rgba      = cv2.cvtColor(resized, cv2.COLOR_BGR2RGBA)

            with self.state.lock:
                self.state.frame_rgba  = rgba
                self.state.hand_detected = hand
                self.state.probs       = probs_all
                self.state.pred        = pred_all
                self.state.latency     = lat_all
                self.state.cond_rgba   = cmap
                self.state.fps         = fps

    def stop(self):
        self._running = False
        self._cam.stop()
        self._ext.close()


# ══════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

def _draw_skeleton(frame, feats, fps: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    if feats is not None:
        pts = feats.reshape(21, 3).numpy()
        p2 = pts[:, :2].copy()
        p2 -= p2.min(0); r = p2.max()
        if r > 0: p2 /= r
        p2 = (p2 * 0.7 + 0.15) * np.array([w, h])
        p2 = p2.astype(int)
        for a, b in CONNECTIONS:
            cv2.line(out, tuple(p2[a]), tuple(p2[b]), (80, 220, 80), 2)
        for i, pt in enumerate(p2):
            cv2.circle(out, tuple(pt), 5,
                       (0, 255, 180) if i == 0 else (0, 200, 255), -1)
    cv2.putText(out, f"{fps:.0f} FPS", (w - 85, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)
    return out


def _conductance_heatmap(model, w: int, h: int) -> np.ndarray:
    layer = model.output_layer
    g_eff = (layer.G_pos - layer.G_neg).detach().cpu().float().numpy()
    mn, mx = g_eff.min(), g_eff.max()
    span = mx - mn if mx - mn > 1e-12 else 1.0
    g8   = ((g_eff - mn) / span * 255).astype(np.uint8)
    rsz  = cv2.resize(g8, (w, h), interpolation=cv2.INTER_NEAREST)
    col  = cv2.applyColorMap(rsz, cv2.COLORMAP_PLASMA)
    return cv2.cvtColor(col, cv2.COLOR_BGR2RGBA)


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
        dpg.create_viewport(title="Memristor Hand-Sign Demo — Live Parallel Comparison",
                            width=WIN_W, height=WIN_H, resizable=False)
        dpg.setup_dearpygui()

        with dpg.texture_registry():
            dpg.add_raw_texture(CAM_W, CAM_H,
                default_value=np.zeros((CAM_H, CAM_W, 4), np.float32).flatten().tolist(),
                format=dpg.mvFormat_Float_rgba, tag="tex_cam")
            dpg.add_raw_texture(CMAP_W, CMAP_H,
                default_value=np.zeros((CMAP_H, CMAP_W, 4), np.float32).flatten().tolist(),
                format=dpg.mvFormat_Float_rgba, tag="tex_cmap")

        # Per-system bar themes
        for sys in SYSTEMS:
            r, g, b = SYS_COLORS[sys]
            with dpg.theme(tag=f"theme_{sys}"):
                with dpg.theme_component(dpg.mvProgressBar):
                    dpg.add_theme_color(dpg.mvThemeCol_PlotHistogram,
                                        (r, g, b, 210))

        with dpg.window(tag="root", no_title_bar=True, no_move=True,
                        no_resize=True, width=WIN_W, height=WIN_H):
            with dpg.group(horizontal=True):
                self._build_left()
                dpg.add_separator()
                self._build_right()

        dpg.set_primary_window("root", True)
        dpg.show_viewport()

    # ── left panel (recording studio) ─────────────────────────────

    def _build_left(self):
        with dpg.child_window(width=LEFT_W, height=WIN_H - 16, border=False, tag="left_panel"):
            dpg.add_text("◉  RECORDING STUDIO", color=(255, 220, 80))
            dpg.add_separator()

            dpg.add_image("tex_cam", width=CAM_W, height=CAM_H, tag="img_cam")
            dpg.add_text("No hand detected", tag="txt_status", color=(100, 180, 255))

            dpg.add_separator()
            dpg.add_text("Train a new sign:", color=(200, 200, 160))
            with dpg.group(horizontal=True):
                dpg.add_input_text(hint="Sign name…", tag="inp_name", width=160)
                dpg.add_button(label="▶  Start", tag="btn_start",
                               callback=self._cb_start)
                dpg.add_button(label="■  Stop", tag="btn_stop",
                               callback=self._cb_stop, show=False)

            dpg.add_progress_bar(default_value=0.0, tag="bar_prog",
                                 width=LEFT_W - 20, height=20, overlay="0 / 80")

            dpg.add_separator()
            dpg.add_text("Device physics (live control):", color=(200, 200, 160))
            dpg.add_slider_float(label="Read noise σ",    tag="sl_read",
                                 default_value=0.01, min_value=0.0, max_value=0.25,
                                 width=300, callback=self._cb_device)
            dpg.add_slider_float(label="Write noise σ",   tag="sl_write",
                                 default_value=0.05, min_value=0.0, max_value=0.35,
                                 width=300, callback=self._cb_device)
            dpg.add_slider_int(label="Conductance levels", tag="sl_levels",
                               default_value=32, min_value=2, max_value=256,
                               width=300, callback=self._cb_device)
            dpg.add_checkbox(label="Apply read noise during inference",
                             tag="chk_noise", default_value=True,
                             callback=self._cb_toggle_noise)

            dpg.add_separator()
            dpg.add_text("Output layer conductance  (G⁺ − G⁻)", color=(200, 200, 160))
            dpg.add_text("Each row = one output class neuron", color=(120, 120, 120))
            dpg.add_image("tex_cmap", width=CMAP_W, height=CMAP_H, tag="img_cmap")
            with dpg.group(horizontal=True):
                dpg.add_text("─ low", color=(80, 80, 220))
                dpg.add_text("                    high ─", color=(220, 80, 80))

    # ── right panel (4-system comparison) ─────────────────────────

    def _build_right(self):
        with dpg.child_window(width=RIGHT_W, height=WIN_H - 16, border=False):
            dpg.add_text("⬡  SYSTEM COMPARISON  (same input, same labels)",
                         color=(255, 220, 80))
            dpg.add_separator()

            # 2×2 confidence grid
            with dpg.group(horizontal=True):
                for col, sys in enumerate(SYSTEMS):
                    with dpg.child_window(
                            tag=f"sys_panel_{sys}",
                            width=(RIGHT_W // 2) - 8,
                            height=280, border=True):
                        color = SYS_COLORS[sys]
                        dpg.add_text(SYS_LABELS[sys], color=color)
                        dpg.add_separator()
                        dpg.add_text("No hand", tag=f"sys_pred_{sys}",
                                     color=(160, 160, 160))
                        dpg.add_separator()
                        for i in range(MAX_CLASSES):
                            show = i < self.state.n_classes
                            with dpg.group(horizontal=True,
                                           tag=f"crow_{sys}_{i}", show=show):
                                nm = self.state.class_names[i] if i < len(self.state.class_names) else f"cls{i}"
                                dpg.add_text(f"{nm:<8}", tag=f"clbl_{sys}_{i}")
                                bar = dpg.add_progress_bar(
                                    default_value=0.0,
                                    tag=f"cbar_{sys}_{i}",
                                    width=CONF_BAR_W, height=20,
                                    overlay="0%",
                                )
                                dpg.bind_item_theme(bar, f"theme_{sys}")
                    if col == 1:
                        pass
            with dpg.group(horizontal=True):
                for sys in SYSTEMS[2:]:
                    with dpg.child_window(
                            tag=f"sys_panel2_{sys}",
                            width=(RIGHT_W // 2) - 8,
                            height=280, border=True):
                        color = SYS_COLORS[sys]
                        dpg.add_text(SYS_LABELS[sys], color=color)
                        dpg.add_separator()
                        dpg.add_text("No hand", tag=f"sys_pred2_{sys}",
                                     color=(160, 160, 160))
                        dpg.add_separator()
                        for i in range(MAX_CLASSES):
                            show = i < self.state.n_classes
                            with dpg.group(horizontal=True,
                                           tag=f"crow2_{sys}_{i}", show=show):
                                nm = self.state.class_names[i] if i < len(self.state.class_names) else f"cls{i}"
                                dpg.add_text(f"{nm:<8}", tag=f"clbl2_{sys}_{i}")
                                bar = dpg.add_progress_bar(
                                    default_value=0.0,
                                    tag=f"cbar2_{sys}_{i}",
                                    width=CONF_BAR_W, height=20,
                                    overlay="0%",
                                )
                                dpg.bind_item_theme(bar, f"theme_{sys}")

            dpg.add_separator()

            # Accuracy plot
            with dpg.child_window(width=RIGHT_W - 8, height=200, border=True):
                dpg.add_text("Accuracy over time  (all 4 systems, live)",
                             color=(200, 200, 160))
                with dpg.plot(height=175, width=RIGHT_W - 20, no_title=True,
                              tag="acc_plot"):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Samples", tag="ax_x")
                    dpg.add_plot_axis(dpg.mvYAxis, label="Acc", tag="ax_y")
                    dpg.set_axis_limits("ax_y", 0.0, 1.0)
                    for sys in SYSTEMS:
                        c = SYS_COLORS[sys]
                        dpg.add_line_series([], [], label=SYS_LABELS[sys],
                                            parent="ax_y", tag=f"acc_{sys}")

            # Latency table
            with dpg.child_window(width=RIGHT_W - 8, height=90, border=True):
                dpg.add_text("Latency  (ms)", color=(200, 200, 160))
                with dpg.table(header_row=True, borders_innerH=True,
                               borders_outerH=True, tag="lat_table"):
                    dpg.add_table_column(label="System",     width_fixed=True, init_width_or_weight=160)
                    dpg.add_table_column(label="Inference",  width_fixed=True, init_width_or_weight=100)
                    dpg.add_table_column(label="Update",     width_fixed=True, init_width_or_weight=100)
                    dpg.add_table_column(label="GPU memory", width_fixed=True, init_width_or_weight=120)
                    for sys in SYSTEMS:
                        with dpg.table_row(tag=f"lat_row_{sys}"):
                            dpg.add_text(SYS_LABELS[sys], color=SYS_COLORS[sys],
                                         tag=f"lat_name_{sys}")
                            dpg.add_text("—", tag=f"lat_inf_{sys}")
                            dpg.add_text("—", tag=f"lat_upd_{sys}")
                            dpg.add_text("—", tag=f"lat_mem_{sys}")

    # ── DPG callbacks ──────────────────────────────────────────────

    def _cb_start(self, *_):
        name = dpg.get_value("inp_name").strip()
        if not name:
            return
        idx = self.state.add_class(name)
        # Expand all models
        for sys, model in self.models.items():
            while idx >= model.n_classes:
                model.add_class()
            # Re-init optimizer for baselines that need it
            trainer = self.trainers[sys]
            if hasattr(trainer, "optimizer"):
                import torch.optim as optim
                trainer.optimizer = optim.Adam(
                    [p for p in model.parameters() if p.requires_grad], lr=0.001
                )
        self.state.training_label = idx
        self.state.n_collected = 0
        dpg.hide_item("btn_start"); dpg.show_item("btn_stop")
        # Show new rows
        for sys in SYSTEMS:
            for tag_prefix in [f"crow_{sys}_{idx}", f"crow2_{sys}_{idx}"]:
                if dpg.does_item_exist(tag_prefix):
                    dpg.configure_item(tag_prefix, show=True)
                    nm = self.state.class_names[idx]
                    for lbl_tag in [f"clbl_{sys}_{idx}", f"clbl2_{sys}_{idx}"]:
                        if dpg.does_item_exist(lbl_tag):
                            dpg.configure_item(lbl_tag, default_value=f"{nm:<8}")

    def _cb_stop(self, *_):
        self.state.training_label = None
        dpg.show_item("btn_start"); dpg.hide_item("btn_stop")

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
            rgba       = self.state.frame_rgba.copy()
            hand       = self.state.hand_detected
            probs_all  = {s: self.state.probs[s].copy() for s in SYSTEMS}
            pred_all   = dict(self.state.pred)
            lat_all    = dict(self.state.latency)
            cmap       = self.state.cond_rgba.copy()
            n_col      = self.state.n_collected
            t_lbl      = self.state.training_label
            names      = list(self.state.class_names)
            acc_hists  = {s: list(self.state.acc_hist[s]) for s in SYSTEMS}

        # Camera
        dpg.set_value("tex_cam",  (rgba.astype(np.float32)/255).flatten().tolist())
        dpg.set_value("tex_cmap", (cmap.astype(np.float32)/255).flatten().tolist())

        # Status
        if hand:
            sys0    = "memristor"
            p0      = pred_all[sys0]
            nm      = names[p0] if p0 < len(names) else f"cls{p0}"
            conf    = float(probs_all[sys0][p0]) if p0 < len(probs_all[sys0]) else 0
            dpg.configure_item("txt_status",
                               default_value=f"✓ Detected  (memristor says: {nm}  {conf:.0%})",
                               color=(80, 255, 140))
        else:
            dpg.configure_item("txt_status",
                               default_value="No hand detected",
                               color=(100, 180, 255))

        # Training progress bar
        if t_lbl is not None:
            ratio = min(n_col / max(self.state.target_samples, 1), 1.0)
            dpg.set_value("bar_prog", ratio)
            dpg.configure_item("bar_prog",
                               overlay=f"{n_col} / {self.state.target_samples}")
            if n_col >= self.state.target_samples:
                self._cb_stop()

        # Confidence bars for top-2 systems (memristor + frozen_lin)
        for sys in SYSTEMS[:2]:
            probs = probs_all[sys]
            pred  = pred_all[sys]
            nm    = names[pred] if pred < len(names) else f"cls{pred}"
            c     = float(probs[pred]) if pred < len(probs) else 0.0
            dpg.configure_item(f"sys_pred_{sys}",
                               default_value=f"{nm}  {c:.0%}" if hand else "No hand")
            for i in range(min(len(names), MAX_CLASSES)):
                v = float(probs[i]) if i < len(probs) else 0.0
                dpg.configure_item(f"cbar_{sys}_{i}",
                                   default_value=v, overlay=f"{v:.0%}")

        # Confidence bars for bottom-2 systems (mlp_sgd + cnn_online)
        for sys in SYSTEMS[2:]:
            probs = probs_all[sys]
            pred  = pred_all[sys]
            nm    = names[pred] if pred < len(names) else f"cls{pred}"
            c     = float(probs[pred]) if pred < len(probs) else 0.0
            dpg.configure_item(f"sys_pred2_{sys}",
                               default_value=f"{nm}  {c:.0%}" if hand else "No hand")
            for i in range(min(len(names), MAX_CLASSES)):
                v = float(probs[i]) if i < len(probs) else 0.0
                dpg.configure_item(f"cbar2_{sys}_{i}",
                                   default_value=v, overlay=f"{v:.0%}")

        # Accuracy plot
        for sys in SYSTEMS:
            hist = acc_hists[sys]
            if hist:
                xs = [h[0] for h in hist]
                ys = [h[1] for h in hist]
                dpg.set_value(f"acc_{sys}", [xs, ys])
        dpg.fit_axis_data("ax_x")

        # Latency table
        for sys in SYSTEMS:
            inf_ms, upd_ms = lat_all[sys]
            mem_mb = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            if dpg.does_item_exist(f"lat_inf_{sys}"):
                dpg.configure_item(f"lat_inf_{sys}",
                                   default_value=f"{inf_ms:.1f} ms")
                dpg.configure_item(f"lat_upd_{sys}",
                                   default_value=f"{upd_ms:.1f} ms")
                dpg.configure_item(f"lat_mem_{sys}",
                                   default_value=f"{mem_mb:.0f} MB")

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
    p.add_argument("--class-names", nargs="+", default=["A", "B", "C", "D", "E"])
    p.add_argument("--model-path", default=None)
    p.add_argument("--target-samples", type=int, default=80)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    with open("config/network.yaml") as f: net_cfg = yaml.safe_load(f)
    with open("config/memristor.yaml") as f: mem_cfg = yaml.safe_load(f)

    n_cls = max(net_cfg["initial_classes"], len(args.class_names))
    dm    = MemristorDeviceModel(mem_cfg)

    # Build all 4 models
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
