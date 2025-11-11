import threading
import time
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json # New import for persistence

# use rich correction/simulation functions
from ColorVisionDemo.color_corrector import (
    simulate_deuteranopia,
    simulate_deuteranomaly,
    daltonize,
    apply_model_to_image,
)

# Lazy TF import & model path candidates
_tf = None
MODEL_CANDIDATES = [
    os.path.join("models", "best_daltonization_model.h5"),
    os.path.join("models", "daltonization_model.h5"),
    os.path.join("models", "model.h5"),
]

# File path constants
TUTORIAL_FLAG_FILE = ".first_run"
CONFIG_FILE = "config.json"


class DaltonizeGUI:
    """
    Tkinter GUI with four panels:
      - Normal (original)
      - Deuteranomaly (simulation)
      - Deuteranopia (simulation)
      - Model-Corrected (correction with intensity 1..100)
    Camera processing runs off the main thread; heavy ops run in a worker thread.
    """

    def __init__(self, root):
        self.root = root
        root.title("Smart Color Aid — Deuteranopia / Daltonize Demo")

        # App state
        self.cap = None
        self.running_camera = False
        self.processing_lock = threading.Lock()
        # lock protecting cached images and sim_nopia to avoid race conditions
        self.cache_lock = threading.Lock()
        self.model = None
        
        # Load persistent settings before building layout
        initial_intensity, initial_use_model = self._load_settings()

        self.use_model = tk.BooleanVar(value=initial_use_model)
        self.intensity = tk.IntVar(value=initial_intensity)  # default intensity 1..100

        # frame-counter + model period for periodic full-model updates during streaming
        self._frame_counter = 0
        self.model_period = 15  # run the heavy model every N frames (tune as needed)

        # Image data buffers (BGR format, NumPy arrays)
        self.orig_bgr = None
        self.sim_anom = None
        self.sim_nopia = None
        self.corrected = None
        # Cached images to enable instant slider response
        self.cached_pred = None  # model prediction (same size as sim_nopia)
        self.cached_daltonized = None  # daltonize fallback

        self._setup_style()
        self._build_layout()
        
        # Check if this is the first run and show the tutorial
        self._check_first_run()

    # --- Persistence Handlers ---
    def _load_settings(self):
        """Loads intensity and model state from config file."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                settings = json.load(f)
                intensity = settings.get("intensity", 75)
                use_model = settings.get("use_model", False)
                return intensity, use_model
        except Exception:
            return 75, False

    def _save_settings(self):
        """Saves current intensity and model state to config file."""
        settings = {
            "intensity": self.intensity.get(),
            "use_model": self.use_model.get()
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            print(f"Error saving settings: {e}")
            
    # --- Style and Layout ---
    def _setup_style(self):
        # simple modern dark-ish ttk styling
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.bg = "#0b1220"
        self.card = "#0f1724"
        self.accent = "#22c55e"
        self.muted = "#9aa7b2"
        self.style.configure(".", background=self.bg, foreground="white")
        self.root.configure(bg=self.bg)

        # --- Button styles: primary (accent) and secondary (subtle) with hover variants ---
        # Primary: accent background, WHITE text normally, black text on hover (as requested)
        # ensure normal and hover preserve identical geometry (font, padding, relief)
        self.style.configure("Primary.TButton",
                             background=self.accent, foreground="#ffffff",
                             padding=6, relief="flat", font=("Segoe UI", 10, "bold"))
        self.style.map("Primary.TButton",
                        background=[("active", "#16a34a"), ("!active", self.accent)])
        self.style.configure("PrimaryHover.TButton",
                             background="#16a34a", foreground="#000000",
                             padding=6, relief="flat", font=("Segoe UI", 10, "bold"))

        # Secondary: darker background, WHITE text normally, black text on hover
        self.style.configure("Secondary.TButton",
                             background="#1f2937", foreground="#ffffff",
                             padding=6, relief="flat", font=("Segoe UI", 10))
        self.style.configure("SecondaryHover.TButton",
                             background="#364152", foreground="#000000",
                             padding=6, relief="flat", font=("Segoe UI", 10))

        # Checkbutton styles (Use Model) - normal and hover (hover text -> black for readability)
        # keep same padding/font for checkbutton hover to avoid layout changes
        self.style.configure("Check.TCheckbutton", background=self.bg, foreground="#ffffff", padding=4, font=("Segoe UI", 10))
        self.style.configure("CheckHover.TCheckbutton", background=self.bg, foreground="#000000", padding=4, font=("Segoe UI", 10))

        # Panel (Labelframe) styles for normal and highlighted states
        # keep PanelHighlight identical to Panel to avoid any visual change on hover
        self.style.configure("Panel.TLabelframe", background=self.card, foreground=self.muted, borderwidth=1, relief="groove")
        self.style.configure("PanelHighlight.TLabelframe", background=self.card, foreground=self.muted, borderwidth=1, relief="groove")
        
        # Color Inspector Style
        self.style.configure("Inspector.TLabel", 
                             background=self.card, 
                             foreground="#ffffff", 
                             font=("Consolas", 9), 
                             padding=6,
                             anchor="w")


    def _build_layout(self):
        # Top controls
        ctrl = ttk.Frame(self.root, padding=8)
        ctrl.pack(fill="x", padx=12, pady=(12, 6))

        # create buttons with styles and keep references so we can bind hover events
        self.load_btn = ttk.Button(ctrl, text="📁 Load Image", command=self.load_image, style="Primary.TButton")
        self.load_btn.pack(side="left", padx=(0, 6))

        # Help button linked to the new show_tutorial method
        self.help_btn = ttk.Button(ctrl, text="❔ Help", command=self.show_tutorial, style="Primary.TButton")
        self.help_btn.pack(side="left", padx=(0, 6))

        self.cam_btn = ttk.Button(ctrl, text="🎥 Start Camera", command=self.toggle_camera, style="Secondary.TButton")
        self.cam_btn.pack(side="left", padx=(0, 6))

        # Use Model as a styled checkbutton so hover can change its text color to readable white
        self.use_model_chk = ttk.Checkbutton(ctrl, text="Use Model", variable=self.use_model,
                                             command=self._on_toggle_model, style="Check.TCheckbutton")
        self.use_model_chk.pack(side="left", padx=(6, 12))

        ttk.Label(ctrl, text="Intensity:", foreground=self.muted).pack(side="left")
        self.intensity_slider = ttk.Scale(ctrl, from_=1, to=100, orient="horizontal", variable=self.intensity, command=self._on_intensity_change)
        self.intensity_slider.pack(side="left", fill="x", expand=True, padx=(6, 6))
        # When user releases the slider, perform a full reprocess (if not streaming) to refresh cached_pred
        # Bind mouse release on the scale to run heavy update once
        try:
            self.intensity_slider.bind("<ButtonRelease-1>", lambda e: self._on_intensity_release())
        except Exception:
            pass
        self.int_label = ttk.Label(ctrl, text=f"{self.intensity.get()}%", foreground=self.muted)
        self.int_label.pack(side="left", padx=(6, 0))
        # Save uses primary style
        self.save_btn = ttk.Button(ctrl, text="💾 Save Corrected", command=self.save_corrected, style="Primary.TButton")
        self.save_btn.pack(side="right", padx=(6, 0))
        
        # New label for color inspection, placed under the main controls
        self.color_info_label = ttk.Label(self.root, 
                                          text="Hover over 'Original Image' for color inspection.",
                                          style="Inspector.TLabel")
        self.color_info_label.pack(fill="x", padx=12, pady=(0, 6))


        # Attach hover handlers to make text readable on hover and optionally highlight a panel container.
        def attach_hover(widget, normal_style=None, hover_style=None, highlight_panel=None, delay=80):
            """
            Debounced hover binder to avoid flicker:
              - schedule a small delayed enter/leave action (ms=delay)
              - cancel the opposite pending job if user moves mouse quickly
              - highlight_panel (ttk.Labelframe) is toggled on enter/leave
            """
            # safely attach attributes for pending job ids
            setattr(widget, "_hover_enter_job", None)
            setattr(widget, "_hover_leave_job", None)

            def do_enter():
                try:
                    if hover_style:
                        widget.configure(style=hover_style)
                except Exception:
                    pass
                if highlight_panel is not None:
                    try:
                        highlight_panel.configure(style="PanelHighlight.TLabelframe")
                    except Exception:
                        pass

            def do_leave():
                try:
                    if normal_style:
                        widget.configure(style=normal_style)
                except Exception:
                    pass
                if highlight_panel is not None:
                    try:
                        highlight_panel.configure(style="Panel.TLabelframe")
                    except Exception:
                        pass

            def on_enter(e):
                # cancel any pending leave job
                job = getattr(widget, "_hover_leave_job", None)
                if job:
                    try:
                        self.root.after_cancel(job)
                    except Exception:
                        pass
                    setattr(widget, "_hover_leave_job", None)
                # schedule enter job
                job = getattr(widget, "_hover_enter_job", None)
                if job:
                    # already scheduled
                    return
                try:
                    j = self.root.after(delay, do_enter)
                    setattr(widget, "_hover_enter_job", j)
                except Exception:
                    # fallback immediate
                    do_enter()

            def on_leave(e):
                # cancel any pending enter job
                job = getattr(widget, "_hover_enter_job", None)
                if job:
                    try:
                        self.root.after_cancel(job)
                    except Exception:
                        pass
                    setattr(widget, "_hover_enter_job", None)
                # schedule leave job
                job = getattr(widget, "_hover_leave_job", None)
                if job:
                    return
                try:
                    j = self.root.after(delay, do_leave)
                    setattr(widget, "_hover_leave_job", j)
                except Exception:
                    do_leave()

            try:
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)
            except Exception:
                pass

        # Four preview panels
        panels = ttk.Frame(self.root, padding=8)
        panels.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        panels.columnconfigure((0, 1, 2, 3), weight=1)

        # Updated Panel Titles for Clarity
        self.panel_normal = ttk.Labelframe(panels, text="Original Image", padding=6, style="Panel.TLabelframe")
        self.panel_anom = ttk.Labelframe(panels, text="Deuteranomaly (Sim)", padding=6, style="Panel.TLabelframe")
        self.panel_nopia = ttk.Labelframe(panels, text="Deuteranopia (Sim)", padding=6, style="Panel.TLabelframe")
        self.panel_model = ttk.Labelframe(panels, text="Model Corrected", padding=6, style="Panel.TLabelframe")

        self.panel_normal.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.panel_anom.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
        self.panel_nopia.grid(row=0, column=2, sticky="nsew", padx=6, pady=6)
        self.panel_model.grid(row=0, column=3, sticky="nsew", padx=6, pady=6)

        # Now that panels exist, attach hover handlers (no panel highlighting to avoid background changes)
        attach_hover(self.load_btn, "Primary.TButton", "PrimaryHover.TButton", delay=80)
        attach_hover(self.help_btn, "Primary.TButton", "PrimaryHover.TButton", delay=80)
        attach_hover(self.save_btn, "Primary.TButton", "PrimaryHover.TButton", delay=80)
        attach_hover(self.cam_btn, "Secondary.TButton", "SecondaryHover.TButton", delay=80)
        attach_hover(self.use_model_chk, "Check.TCheckbutton", "CheckHover.TCheckbutton", delay=80)

        # Image labels (will hold ImageTk)
        self.lbl_normal = tk.Label(self.panel_normal, text="No image", bg=self.card, fg=self.muted)
        self.lbl_anom = tk.Label(self.panel_anom, text="No image", bg=self.card, fg=self.muted)
        self.lbl_nopia = tk.Label(self.panel_nopia, text="No image", bg=self.card, fg=self.muted)
        self.lbl_model = tk.Label(self.panel_model, text="No image", bg=self.card, fg=self.muted)

        for lbl in (self.lbl_normal, self.lbl_anom, self.lbl_nopia, self.lbl_model):
            lbl.pack(fill="both", expand=True)

        # BIND THE COLOR INSPECTOR HOVER EVENT
        self.lbl_normal.bind("<Motion>", self._on_color_inspect)
        self.lbl_normal.bind("<Leave>", self._on_color_inspect_leave)

        # Status bar
        self.status = ttk.Label(self.root, text="Ready", anchor="w")
        self.status.pack(fill="x", padx=12, pady=(0, 12))

    # --- Color Inspector Logic ---
    def _on_color_inspect_leave(self, event):
        """Resets the color inspection label when the mouse leaves the panel."""
        self.color_info_label.config(text="Hover over 'Original Image' for color inspection.")

    def _on_color_inspect(self, event):
        """
        Handles the mouse motion event to inspect the color at the cursor.
        event.x and event.y are coordinates relative to the tk.Label widget.
        """
        if self.orig_bgr is None:
            return

        # 1. Get current image dimensions
        # Get the dimensions of the displayed photo image (ImgTk is stored on the label)
        imgtk = self.lbl_normal.imgtk
        if not imgtk:
            return
            
        display_w = imgtk.width()
        display_h = imgtk.height()
        
        # 2. Get original image dimensions
        orig_h, orig_w = self.orig_bgr.shape[:2]
        
        # 3. Calculate scaling factor used in _set_img_on_label
        # Note: we need to use the actual size of the image displayed on the canvas, 
        # but since we are scaling proportionally to a max_dim=360, we can use the ratio of original/displayed
        
        # 4. Map event coordinates (x, y) back to original image (orig_bgr) indices
        # Ensure we don't divide by zero and handle edges
        if display_w == 0 or display_h == 0:
            return
            
        # Scale X and Y from label coordinates to original image coordinates
        x_scaled = int(event.x * (orig_w / display_w))
        y_scaled = int(event.y * (orig_h / display_h))
        
        # Clamp coordinates to bounds
        x_idx = np.clip(x_scaled, 0, orig_w - 1)
        y_idx = np.clip(y_scaled, 0, orig_h - 1)

        # 5. Extract colors from all buffers (read under lock for thread safety)
        with self.cache_lock:
            if self.orig_bgr is None or self.sim_anom is None or self.sim_nopia is None or self.corrected is None:
                return

            # Note: OpenCV uses BGR format
            orig_bgr = self.orig_bgr[y_idx, x_idx]
            anom_bgr = self.sim_anom[y_idx, x_idx]
            nopia_bgr = self.sim_nopia[y_idx, x_idx]
            corrected_bgr = self.corrected[y_idx, x_idx]

        # Function to format BGR array to Hex string
        def bgr_to_hex(bgr):
            r, g, b = bgr[2], bgr[1], bgr[0]
            return f"#{r:02X}{g:02X}{b:02X}"

        # Function to format BGR array to RGB string
        def bgr_to_rgb(bgr):
            r, g, b = bgr[2], bgr[1], bgr[0]
            return f"RGB({r}, {g}, {b})"

        # 6. Format output string
        output = (
            f"Pixel [{x_idx}, {y_idx}] | "
            f"Original: {bgr_to_hex(orig_bgr)} ({bgr_to_rgb(orig_bgr)}) | "
            f"Anomaly: {bgr_to_hex(anom_bgr)} | "
            f"Nopia: {bgr_to_hex(nopia_bgr)} | "
            f"Corrected: {bgr_to_hex(corrected_bgr)}"
        )
        
        # 7. Update label
        self.color_info_label.config(text=output)


    # ---------- First-Run Check ----------
    def _check_first_run(self):
        """Checks for the flag file and shows tutorial if it doesn't exist."""
        if not os.path.exists(TUTORIAL_FLAG_FILE):
            # Use root.after to ensure the main window is fully rendered before opening the tutorial modal
            self.root.after(200, self.show_tutorial)
            # Create the flag file so it doesn't run again
            try:
                with open(TUTORIAL_FLAG_FILE, 'w') as f:
                    f.write("Tutorial viewed.")
            except Exception:
                # If writing the file fails, just proceed without automatic showing next time
                pass

    # ---------- UI actions ----------
    def _on_intensity_change(self, _=None):
        self.int_label.config(text=f"{self.intensity.get()}%")
        # Perform an immediate, lightweight update of the Model-Corrected preview using cached images
        # This keeps slider movement smooth and responsive
        self._quick_blend_update()

    def _on_intensity_release(self):
        """Called when user releases the slider: refresh full processing for highest-quality result
        (only when not streaming to avoid blocking camera preview)."""
        if not self.running_camera and self.orig_bgr is not None:
            # full reprocess to refresh cached_pred (may be expensive)
            self.status.config(text="Reprocessing with updated intensity...")
            self.root.update_idletasks()
            # Run synchronous processing (non-fast) so model predict runs and cached_pred updates
            self._process_and_update(self.orig_bgr.copy(), fast=False)
            self.status.config(text="Ready")

    def _quick_blend_update(self):
        """Fast main-thread blending using cached images to instantly update Model-Corrected preview."""
        if self.sim_nopia is None and self.orig_bgr is None:
            return
        alpha = float(np.clip(self.intensity.get(), 1, 100)) / 100.0

        # Read caches under lock to avoid races with background worker
        with self.cache_lock:
            sim = self.sim_nopia.copy() if (self.sim_nopia is not None) else None
            pred_cached = self.cached_pred.copy() if (self.cached_pred is not None) else None
            dalt_cached = self.cached_daltonized.copy() if (self.cached_daltonized is not None) else None

        # Prefer to blend using cached model prediction if available
        if self.use_model.get() and (pred_cached is not None) and (sim is not None):
            try:
                blended = self._blend_chroma(sim, pred_cached, alpha)
                self._set_img_on_label(self.lbl_model, blended)
                return
            except Exception:
                pass

        # If model prediction is not cached but daltonize fallback exists, blend with it
        if self.use_model.get() and (dalt_cached is not None) and (sim is not None):
            try:
                blended = self._blend_chroma(sim, dalt_cached, alpha)
                self._set_img_on_label(self.lbl_model, blended)
                return
            except Exception:
                pass

        # Otherwise, simple blend between normal and simulated
        base = self.orig_bgr if (self.orig_bgr is not None) else sim
        if base is None or sim is None:
            return
        try:
            blended = self._blend_chroma(base, sim, alpha)
            self._set_img_on_label(self.lbl_model, blended)
        except Exception:
            blended = cv2.addWeighted(base, 1 - alpha, sim, alpha, 0)
            self._set_img_on_label(self.lbl_model, blended)

    def _on_toggle_model(self):
        if self.use_model.get():
            self.status.config(text="Model enabled — loading if required...")
            self.root.update_idletasks()
            self._ensure_model_loaded()
        else:
            self.status.config(text="Model disabled")

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tiff")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot open image")
            return
        # If camera was running, stop it to avoid concurrent updates between camera and a loaded image
        if self.running_camera:
            self._stop_camera()
        self.orig_bgr = img
        self._process_and_update(img.copy(), fast=False)
        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def toggle_camera(self):
        if self.running_camera:
            self._stop_camera()
        else:
            self._start_camera()

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        # conservative resolution to reduce processing load
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        except Exception:
            pass
        if not (self.cap and self.cap.isOpened()):
            messagebox.showerror("Error", "Failed to open camera")
            return
        self.running_camera = True
        self.cam_btn.config(text="⏹ Stop Camera")
        self.status.config(text="Camera running")
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _stop_camera(self):
        self.running_camera = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cam_btn.config(text="🎥 Start Camera")
        self.status.config(text="Camera stopped")

    def _camera_loop(self):
        # read frames continuously; dispatch worker for processing with non-blocking lock
        while self.running_camera:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            frame = cv2.flip(frame, 1)  # mirror for user-friendly preview
            self.orig_bgr = frame

            # increment frame counter
            self._frame_counter += 1

            # decide whether to run a full (slow) processing or fast processing for this frame
            # Full processing (fast=False) will run model prediction when available.
            # Run full processing periodically when Use Model is enabled and model is loaded.
            if self.use_model.get() and (self.model is not None):
                do_full = (self._frame_counter % self.model_period) == 0
            else:
                do_full = False

            fast_flag = not do_full  # compute_all uses fast=True to skip heavy model prediction

            # Try to acquire lock (non-blocking); if busy, drop frame to avoid queueing
            if self.processing_lock.acquire(False):
                threading.Thread(target=self._worker_process, args=(frame.copy(), fast_flag), daemon=True).start()
            else:
                # drop frame
                pass
            time.sleep(0.02)

    def _worker_process(self, frame, fast=True):
        try:
            sim_anom, sim_nopia, corrected = self._compute_all(frame, fast=fast)

            # schedule UI update on main thread
            # update quick caches (daltonize fallback typically used for fast)
            try:
                with self.cache_lock:
                    self.cached_daltonized = daltonize(sim_nopia.copy(), enhance_luminance=False)
                    # store a safe copy of sim_nopia for quick blends
                    self.sim_nopia = sim_nopia.copy()
            except Exception:
                with self.cache_lock:
                    self.cached_daltonized = corrected.copy()
                    self.sim_nopia = sim_nopia.copy()

            self.root.after(0, lambda: self._update_all_previews(frame, sim_anom, sim_nopia, corrected))
        finally:
            try:
                self.processing_lock.release()
            except Exception:
                pass

    def _compute_all(self, bgr, fast=False):
        """
        Return (sim_deuteranomaly, sim_deuteranopia, model_corrected)
        - fast=True: avoid heavy model.predict; use daltonize fallback for corrected.
        - intensity values 1..100 control blending strength.
        """
        # always produce simulations using color_corrector functions
        try:
            sim_anom = simulate_deuteranomaly(bgr.copy(), severity=0.6)
        except Exception:
            sim_anom = bgr.copy()

        try:
            sim_nopia = simulate_deuteranopia(bgr.copy())
        except Exception:
            sim_nopia = bgr.copy()

        # compute corrected
        alpha = float(np.clip(self.intensity.get(), 1, 100)) / 100.0

        # prefer model if requested and available, otherwise daltonize fallback
        if self.use_model.get():
            # if model not loaded, attempt to lazy-load
            if self.model is None and not fast:
                self._ensure_model_loaded()
            # If model exists and we are allowed to run heavy ops (fast==False) then produce a real prediction.
            # Otherwise fall back to daltonize and cache it for quick slider blends.
            if self.model is not None and not fast:
                try:
                    pred = apply_model_to_image(sim_nopia, self.model)
                    # blend chroma channels in LAB according to intensity
                    # cache full prediction for responsive slider blending
                    with self.cache_lock:
                        self.cached_pred = pred.copy()
                        # also ensure sim_nopia cached for slider blends
                        self.sim_nopia = sim_nopia.copy()
                    lab_sim = cv2.cvtColor(sim_nopia, cv2.COLOR_BGR2LAB).astype(np.float32)
                    lab_pred = cv2.cvtColor(pred, cv2.COLOR_BGR2LAB).astype(np.float32)
                    blended = lab_sim.copy()
                    blended[..., 1] = lab_sim[..., 1] * (1 - alpha) + lab_pred[..., 1] * alpha
                    blended[..., 2] = lab_sim[..., 2] * (1 - alpha) + lab_pred[..., 2] * alpha
                    blended = np.clip(blended, 0, 255).astype(np.uint8)
                    corrected = cv2.cvtColor(blended, cv2.COLOR_LAB2BGR)
                except Exception:
                    corrected = daltonize(sim_nopia.copy(), enhance_luminance=True)
                    with self.cache_lock:
                        self.cached_daltonized = corrected.copy()
                        self.sim_nopia = sim_nopia.copy()
            else:
                # fast path or model missing: daltonize fallback
                corrected = daltonize(sim_nopia.copy(), enhance_luminance=True)
                # cache fallback for slider blending
                with self.cache_lock:
                    self.cached_daltonized = corrected.copy()
                    self.sim_nopia = sim_nopia.copy()
                # blend between sim_nopia and daltonize according to alpha
                corrected = self._blend_chroma(sim_nopia, corrected, alpha)
        else:
            # No model: blend original and simulated according to intensity to preserve context
            corrected = self._blend_chroma(bgr.copy(), sim_nopia.copy(), alpha)

        return sim_anom, sim_nopia, corrected

    def _blend_chroma(self, base_bgr, target_bgr, alpha):
        """
        Blend chroma channels (a,b in LAB) between base and target according to alpha,
        preserve base luminance for clarity.
        """
        try:
            lab_base = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            blended = lab_base.copy()
            blended[..., 1] = lab_base[..., 1] * (1 - alpha) + lab_target[..., 1] * alpha
            blended[..., 2] = lab_base[..., 2] * (1 - alpha) + lab_target[..., 2] * alpha
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            return cv2.cvtColor(blended, cv2.COLOR_LAB2BGR)
        except Exception:
            # fallback to simple weighted blend
            try:
                return cv2.addWeighted(base_bgr, 1 - alpha, target_bgr, alpha, 0)
            except Exception:
                return target_bgr

    def _update_all_previews(self, normal_bgr, sim_anom, sim_nopia, corrected):
        # Update internal and UI images (called on main thread)
        self.sim_anom = sim_anom
        self.sim_nopia = sim_nopia
        self.corrected = corrected
        self._set_img_on_label(self.lbl_normal, normal_bgr)
        self._set_img_on_label(self.lbl_anom, sim_anom)
        self._set_img_on_label(self.lbl_nopia, sim_nopia)
        self._set_img_on_label(self.lbl_model, corrected)

    def _set_img_on_label(self, label_widget, bgr):
        # Downscale to fit label region while preserving aspect ratio
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        max_dim = 360
        scale = min(1.0, max_dim / max(w, h))
        if scale < 1.0:
            disp = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp = bgr
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)
        label_widget.imgtk = imgtk  # keep reference
        label_widget.config(image=imgtk, text="")

    def _ensure_model_loaded(self):
        global _tf
        if self.model is not None:
            return
        try:
            import tensorflow as tf
            _tf = tf
        except Exception as e:
            messagebox.showerror("TensorFlow load error", f"Could not import tensorflow:\n{e}")
            self.use_model.set(False)
            return
        # find model file
        found = None
        for p in MODEL_CANDIDATES:
            if os.path.exists(p):
                found = p
                break
        if not found:
            messagebox.showwarning("Model not found", "No model file found in models/*.h5")
            self.use_model.set(False)
            return
        try:
            self.model = _tf.keras.models.load_model(found, compile=False)
            self.status.config(text=f"Model loaded: {os.path.basename(found)}")
        except Exception as e:
            messagebox.showerror("Model load error", f"Failed to load model:\n{e}")
            self.use_model.set(False)

    def save_corrected(self):
        if self.corrected is None:
            messagebox.showinfo("Nothing to save", "No corrected image available.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if not path:
            return
        ok = cv2.imwrite(path, self.corrected)
        if ok:
            self.status.config(text=f"Saved: {os.path.basename(path)}")
        else:
            messagebox.showerror("Save failed", "Could not save image")

    def _process_and_update(self, frame, fast=False):
        # Synchronous processing for static images (or manual reprocess)
        if self.processing_lock.acquire(False):
            try:
                sim_anom, sim_nopia, corrected = self._compute_all(frame, fast=fast)
                self._update_all_previews(frame, sim_anom, sim_nopia, corrected)
            finally:
                try:
                    self.processing_lock.release()
                except Exception:
                    pass
        else:
            # if busy, schedule a short retry
            self.root.after(100, lambda: self._process_and_update(frame, fast=fast))

    def show_tutorial(self):
        """Open a modal dialog with a clear overview and instructions (structured and readable)."""
        # Build a modal with tagged formatting for headings and body text
        win = tk.Toplevel(self.root)
        win.title("Simulation Overview")
        win.transient(self.root)
        win.grab_set()
        win.geometry("760x520")

        header = ttk.Label(win, text="Smart Color Aid — Overview", font=("Segoe UI", 14, "bold"))
        header.pack(anchor="w", padx=12, pady=(12, 6))

        frame = ttk.Frame(win)
        frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        text = tk.Text(frame, wrap="word", bg=self.card, fg="#e6eef8", bd=0, relief="flat")
        text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=text.yview)
        sb.pack(side="right", fill="y")
        text.configure(yscrollcommand=sb.set, padx=10, pady=10)

        # Tags for structured formatting
        text.tag_configure("title", font=("Segoe UI", 12, "bold"), foreground=self.accent, spacing3=6)
        text.tag_configure("section", font=("Segoe UI", 10, "bold"), foreground="#e6eef8", spacing1=6, spacing3=4)
        text.tag_configure("step_title", font=("Segoe UI", 10, "bold"), foreground=self.accent, lmargin1=10, spacing1=4)
        text.tag_configure("body", font=("Segoe UI", 10), foreground="#cbd5e1", lmargin1=10, lmargin2=10, spacing1=2, spacing3=6)
        text.tag_configure("bullet", font=("Segoe UI", 10), foreground="#cbd5e1", lmargin1=20, lmargin2=40)

        # Insert structured content (Tutorial focus)
        text.insert("end", "💡 Welcome to Smart Color Aid!\n", "title")
        text.insert("end", "This quick tour explains the four panels you see in the main window and how the tool helps with color vision deficiency.\n\n", "body")

        text.insert("end", "Panel 1: Original Image\n", "step_title")
        text.insert("end", "• This is the image you loaded or the live camera feed, displayed without any modification. It serves as your primary reference point.\n\n", "bullet")

        text.insert("end", "Panel 2: Deuteranomaly (Sim)\n", "step_title")
        text.insert("end", "• This simulates the perception of **Deuteranomaly**, the most common type of red-green color blindness. Colors often appear subdued or 'muddy'.\n\n", "bullet")

        text.insert("end", "Panel 3: Deuteranopia (Sim)\n", "step_title")
        text.insert("end", "• This simulates **Deuteranopia**, a more severe form of red-green color blindness where the primary red-sensing cones are completely missing.\n\n", "bullet")

        text.insert("end", "Panel 4: Model Corrected\n", "step_title")
        text.insert("end", "• This panel shows the corrected image. Our algorithm adjusts the color hues (chroma) to maximize the distinction between colors that appear similar in the simulated views.\n", "bullet")
        text.insert("end", "• Use the **Intensity** slider at the top to control the strength of this correction (blending strength) for the best balance.\n\n", "bullet")

        text.insert("end", "Controls & Performance\n", "section")
        text.insert("end", "• Use Model Checkbox: Toggles the use of the powerful, trained neural network model. If the model file is missing, an algorithmic fallback (Daltonize) is used instead.\n", "bullet")
        text.insert("end", "• Performance: Live camera processing and model predictions run in background threads, scheduled periodically to ensure the application remains highly responsive.\n", "bullet")


        text.configure(state="disabled")

        close_btn = ttk.Button(win, text="Close", command=win.destroy, style="Primary.TButton")
        close_btn.pack(pady=(0, 12))
        try:
            close_btn.bind("<Enter>", lambda e: close_btn.configure(style="PrimaryHover.TButton"))
            close_btn.bind("<Leave>", lambda e: close_btn.configure(style="Primary.TButton"))
        except Exception:
            pass


def main():
    root = tk.Tk()
    app = DaltonizeGUI(root)
    # Modified protocol to save settings before closing
    def on_closing():
        app._stop_camera()
        app._save_settings()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.geometry("1400x520")
    root.mainloop()


if __name__ == "__main__":
    main()