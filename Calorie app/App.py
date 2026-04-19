import tkinter as tk
from tkinter import filedialog, ttk
import json, os, threading
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter

# ── Load your model files ──────────────────────────────────────────────────────
import tensorflow as tf

MODEL_PATH   = "best_model_v2.keras"
CLASSES_PATH = "class_names.json"
CALORIES_PATH= "calorie_table.json"

# ── Colours ────────────────────────────────────────────────────────────────────
BG       = "#f3f4f6"
CARD     = "#ffffff"
BORDER   = "#d1d5db"
ORANGE   = "#fb923c"
RED      = "#ef4444"
GREEN    = "#34d399"
BLUE     = "#60a5fa"
PURPLE   = "#c084fc"
MUTED    = "#475569"
TEXT     = "#0f172a"
SUBTEXT  = "#64748b"

FONT_H1  = ("Segoe UI", 22, "bold")
FONT_H2  = ("Segoe UI", 14, "bold")
FONT_H3  = ("Segoe UI", 11, "bold")
FONT_B   = ("Segoe UI", 10)
FONT_S   = ("Segoe UI", 9)
FONT_NUM = ("Segoe UI", 32, "bold")

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def resolve_names(class_names, n):
    if isinstance(class_names, list):
        return class_names
    if isinstance(class_names, dict):
        if "0" in class_names or 0 in class_names:
            return [class_names.get(str(i), class_names.get(i, f"Class {i}")) for i in range(n)]
        inv = {v: k for k, v in class_names.items()}
        return [inv.get(i, f"Class {i}") for i in range(len(inv))]
    return [f"Class {i}" for i in range(n)]

def estimate_macros(cal_per_100g, grams):
    f = grams / 100
    total = cal_per_100g * f
    return {
        "Calories": (int(total),   "kcal", ORANGE),
        "Protein":  (round(cal_per_100g*0.08*f/4, 1), "g", GREEN),
        "Carbs":    (round(cal_per_100g*0.55*f/4, 1), "g", BLUE),
        "Fat":      (round(cal_per_100g*0.20*f/9, 1), "g", RED),
        "Fiber":    (round(cal_per_100g*0.04*f,   1), "g", PURPLE),
    }

def make_rounded_image(img, size=(320, 320), radius=20):
    img = img.copy()
    img.thumbnail(size, Image.LANCZOS)
    bg = Image.new("RGBA", size, (26, 26, 26, 255))
    offset = ((size[0]-img.width)//2, (size[1]-img.height)//2)
    bg.paste(img.convert("RGBA"), offset)
    mask = Image.new("L", size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0,0,size[0]-1,size[1]-1], radius=radius, fill=255)
    bg.putalpha(mask)
    return bg


# ══════════════════════════════════════════════════════════════════════════════
class CalorieApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(" Indian Food Calorie Estimator")
        self.geometry("960x700")
        self.minsize(860, 620)
        self.configure(bg=BG)
        self.resizable(True, True)

        # State
        self.model        = None
        self.class_names  = []
        self.calorie_table= {}
        self.img_path     = None
        self.serving_var  = tk.IntVar(value=250)
        self.search_var   = tk.StringVar()
        self.result_data  = None   # store last prediction

        self._load_assets()
        self._build_ui()

    # ── Asset loading ──────────────────────────────────────────────────────────
    def _load_assets(self):
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            raw_names  = load_json(CLASSES_PATH)
            n = self.model.output_shape[-1]
            self.class_names   = resolve_names(raw_names, n)
            self.calorie_table = load_json(CALORIES_PATH)
            self._status("✅ Model loaded successfully", GREEN)
        except Exception as e:
            self._status(f"⚠️  {e}", RED)

    # ── UI build ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(hdr, text="🍛 Indian Food Calorie Estimator",
                 font=FONT_H1, fg=ORANGE, bg=BG).pack(side="left")

        self.status_lbl = tk.Label(hdr, text="", font=FONT_S, fg=GREEN, bg=BG)
        self.status_lbl.pack(side="right", padx=4)

        sep = tk.Frame(self, bg=BORDER, height=1)
        sep.pack(fill="x", padx=24, pady=6)

        # Body — two columns
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=24, pady=8)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        self._build_left(body)
        self._build_right(body)

    def _build_left(self, parent):
        lf = tk.Frame(parent, bg=BG)
        lf.grid(row=0, column=0, sticky="nsew", padx=(0,12))
        lf.rowconfigure(1, weight=1)

        # Upload card
        upload_card = self._card(lf)
        upload_card.pack(fill="x", pady=(0,10))

        tk.Label(upload_card, text="📤  Upload Food Image",
                 font=FONT_H3, fg=TEXT, bg=CARD).pack(anchor="w", pady=(0,10))

        # Image preview canvas
        self.canvas = tk.Canvas(upload_card, width=300, height=250,
                                bg="#111", highlightthickness=1,
                                highlightbackground=BORDER)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", lambda e: self._pick_image())
        self._draw_placeholder()

        btn_frame = tk.Frame(upload_card, bg=CARD)
        btn_frame.pack(fill="x", pady=(10,0))

        self._btn(btn_frame, "📁  Browse Image", self._pick_image, ORANGE).pack(fill="x", pady=2)
        self._btn(btn_frame, "🔍  Analyse", self._run_prediction, "#16a34a").pack(fill="x", pady=2)
        self._btn(btn_frame, "✖  Clear", self._reset_app, RED).pack(fill="x", pady=2)

        # Manual lookup card
        lookup_card = self._card(lf)
        lookup_card.pack(fill="x", pady=(0,10))

        tk.Label(lookup_card, text="🔎  Search Food by Name",
                 font=FONT_H3, fg=TEXT, bg=CARD).pack(anchor="w", pady=(0,8))
        search_row = tk.Frame(lookup_card, bg=CARD)
        search_row.pack(fill="x", pady=(0,8))

        self.lookup_entry = tk.Entry(search_row, textvariable=self.search_var,
                                     font=FONT_B, bg="#111", fg="#ffffff",
                                     insertbackground="#ffffff", relief="flat")
        self.lookup_entry.pack(side="left", fill="x", expand=True, padx=(0,8), pady=2)
        self.lookup_entry.bind("<Return>", lambda e: self._run_lookup())
        self._btn(search_row, "Search", self._run_lookup, BLUE).pack(side="right")

        self.lookup_lbl = tk.Label(lookup_card,
                                   text="Enter a food name like aloo_gobi or jalebi",
                                   font=FONT_S, fg=SUBTEXT, bg=CARD, wraplength=260, justify="left")
        self.lookup_lbl.pack(anchor="w")

        # Serving slider
        srv_card = self._card(lf)
        srv_card.pack(fill="x", pady=(0,10))

        tk.Label(srv_card, text="🍽️  Serving Size", font=FONT_H3, fg=TEXT, bg=CARD).pack(anchor="w")
        self.serving_disp = tk.Label(srv_card, text="250 g",
                                     font=("Segoe UI", 16, "bold"), fg=ORANGE, bg=CARD)
        self.serving_disp.pack(anchor="w")

        sl = ttk.Scale(srv_card, from_=50, to=500, variable=self.serving_var,
                       orient="horizontal", command=self._on_serving)
        sl.pack(fill="x", pady=4)

        tk.Label(srv_card, text="50g ←──────────────→ 500g",
                 font=FONT_S, fg=MUTED, bg=CARD).pack()

        # Status bar
        self.file_lbl = tk.Label(lf, text="No file selected",
                                 font=FONT_S, fg=MUTED, bg=BG, anchor="w")
        self.file_lbl.pack(fill="x", pady=4)

    def _build_right(self, parent):
        rf = tk.Frame(parent, bg=BG)
        rf.grid(row=0, column=1, sticky="nsew")
        rf.rowconfigure(0, weight=1)

        # Placeholder welcome
        self.right_placeholder = tk.Frame(rf, bg=CARD, bd=0)
        self.right_placeholder.pack(fill="both", expand=True, pady=0)

        for icon, title, desc in [
            ("📸", "Upload a Photo",    "Click 'Browse Image' or tap the preview"),
            ("🧠", "AI Detection",      "MobileNetV2 identifies food from 80 classes"),
            ("📊", "Get Calories",      "Instant calorie + macro estimate"),
        ]:
            row = tk.Frame(self.right_placeholder, bg=CARD)
            row.pack(fill="x", padx=20, pady=14)
            tk.Label(row, text=icon, font=("Segoe UI", 28), bg=CARD).pack(side="left", padx=(0,12))
            col = tk.Frame(row, bg=CARD)
            col.pack(side="left")
            tk.Label(col, text=title, font=FONT_H3, fg=TEXT, bg=CARD).pack(anchor="w")
            tk.Label(col, text=desc,  font=FONT_S,  fg=MUTED, bg=CARD).pack(anchor="w")

        # Results frame (hidden until prediction)
        self.results_frame = tk.Frame(rf, bg=BG)

    # ── Widgets helpers ────────────────────────────────────────────────────────
    def _card(self, parent, **kw):
        f = tk.Frame(parent, bg=CARD, bd=0, relief="flat",
                     highlightthickness=1, highlightbackground=BORDER, **kw)
        inner = tk.Frame(f, bg=CARD, padx=14, pady=12)
        inner.pack(fill="both", expand=True)
        return f

    def _btn(self, parent, text, cmd, color):
        c = tk.Canvas(parent, height=44, bg=CARD, bd=0,
                      highlightthickness=0, relief="flat", cursor="hand2")
        def redraw(event=None):
            c.delete("all")
            w = max(c.winfo_width(), 120)
            h = max(c.winfo_height(), 44)
            r = 16
            x1, y1, x2, y2 = 0, 0, w, h
            c.create_rectangle(x1+r, y1, x2-r, y2, fill=color, width=0)
            c.create_rectangle(x1, y1+r, x2, y2-r, fill=color, width=0)
            c.create_oval(x1, y1, x1+2*r, y1+2*r, fill=color, outline="")
            c.create_oval(x2-2*r, y1, x2, y1+2*r, fill=color, outline="")
            c.create_oval(x1, y2-2*r, x1+2*r, y2, fill=color, outline="")
            c.create_oval(x2-2*r, y2-2*r, x2, y2, fill=color, outline="")
            c.create_text(w/2, h/2, text=text, fill="white", font=FONT_B)
        c.bind("<Configure>", redraw)
        c.bind("<Button-1>", lambda e: cmd())
        return c

    def _draw_placeholder(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 300, 250, fill="#eff6ff", outline="")
        radius = 20
        x1, y1, x2, y2 = 12, 12, 288, 238
        self.canvas.create_rectangle(x1+radius, y1, x2-radius, y2, fill="#ffffff", width=0)
        self.canvas.create_rectangle(x1, y1+radius, x2, y2-radius, fill="#ffffff", width=0)
        self.canvas.create_oval(x1, y1, x1+2*radius, y1+2*radius, fill="#ffffff", outline="")
        self.canvas.create_oval(x2-2*radius, y1, x2, y1+2*radius, fill="#ffffff", outline="")
        self.canvas.create_oval(x1, y2-2*radius, x1+2*radius, y2, fill="#ffffff", outline="")
        self.canvas.create_oval(x2-2*radius, y2-2*radius, x2, y2, fill="#ffffff", outline="")
        self.canvas.create_text(150, 105, text="🖼️", font=("Segoe UI", 40), fill=SUBTEXT)
        self.canvas.create_text(150, 155, text="Click to browse image",
                                font=FONT_S, fill=TEXT)
        self.canvas.create_text(150, 175, text="JPG · PNG · WEBP",
                                font=FONT_S, fill=TEXT)

    def _status(self, msg, color=SUBTEXT):
        try:
            self.status_lbl.config(text=msg, fg=color)
        except:
            pass

    def _find_food_match(self, query):
        normalized = query.strip().lower().replace(" ", "_").replace("-", "_")
        normalized = "".join(ch for ch in normalized if ch.isalnum() or ch == "_")
        normalized = normalized.strip("_")
        if not normalized:
            return None, None
        if normalized in self.calorie_table:
            return normalized, self.calorie_table[normalized]

        for key in self.calorie_table:
            if normalized == key:
                return key, self.calorie_table[key]
        for key in self.calorie_table:
            if normalized in key:
                return key, self.calorie_table[key]
        for key in self.calorie_table:
            if key.startswith(normalized) or normalized.startswith(key):
                return key, self.calorie_table[key]
        return None, None

    def _reset_app(self):
        self.img_path = None
        self.file_lbl.config(text="No file selected", fg=MUTED)
        self.search_var.set("")
        self.lookup_lbl.config(text="Enter a food name like aloo_gobi or jalebi", fg=SUBTEXT)
        self.serving_var.set(250)
        self.serving_disp.config(text="250 g")
        self.result_data = None
        self._draw_placeholder()
        for w in self.results_frame.winfo_children():
            w.destroy()
        self.results_frame.pack_forget()
        if not self.right_placeholder.winfo_ismapped():
            self.right_placeholder.pack(fill="both", expand=True, pady=0)
        self._status("Ready", GREEN)

    def _run_lookup(self):
        query = self.search_var.get().strip()
        if not query:
            self._status("⚠️  Enter a food item to lookup", RED)
            self.lookup_lbl.config(text="Type a food item name and press Search.", fg=SUBTEXT)
            return

        name, calories = self._find_food_match(query)
        if name is None:
            self._status(f"⚠️  No match for '{query}'", RED)
            self.lookup_lbl.config(text=f"No match found for '{query}'. Try another food name.", fg=RED)
            return

        display_name = name.replace("_", " ").title()
        self.lookup_lbl.config(text=f"{display_name}: {calories} kcal per 100g", fg=TEXT)
        self._status(f"✅  Found '{display_name}'", GREEN)

        self.result_data = ([name, "Unknown", "Unknown"], [1.0, 0.0, 0.0], [calories, 0, 0])
        self._render_results(self.result_data[0], self.result_data[1], self.result_data[2])

    # ── Events ─────────────────────────────────────────────────────────────────
    def _on_serving(self, _=None):
        g = self.serving_var.get()
        self.serving_disp.config(text=f"{g} g")
        if self.result_data:
            self._render_results(*self.result_data)

    def _pick_image(self):
        path = filedialog.askopenfilename(
            title="Select Food Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if not path:
            return
        self.img_path = path
        self.file_lbl.config(text=f"📂 {os.path.basename(path)}", fg=SUBTEXT)

        img = Image.open(path)
        rounded = make_rounded_image(img, size=(300, 250), radius=14)
        self._tk_img = ImageTk.PhotoImage(rounded)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    def _run_prediction(self):
        if not self.img_path:
            self._status("⚠️  Please select an image first", RED)
            return
        if self.model is None:
            self._status("⚠️  Model not loaded", RED)
            return
        self._status("🧠  Analysing…", BLUE)
        threading.Thread(target=self._predict_thread, daemon=True).start()

    def _predict_thread(self):
        try:
            img    = Image.open(self.img_path)
            tensor = preprocess(img)
            preds  = self.model.predict(tensor, verbose=0)[0]
            top3   = np.argsort(preds)[::-1][:3]
            names  = [self.class_names[i] for i in top3]
            confs  = [float(preds[i]) for i in top3]
            cals   = [self.calorie_table.get(n, self.calorie_table.get(n.lower(), 200)) for n in names]
            self.result_data = (names, confs, cals)
            self.after(0, self._render_results, names, confs, cals)
            self.after(0, self._status, "✅  Done!", GREEN)
        except Exception as e:
            self.after(0, self._status, f"❌  {e}", RED)

    # ── Results rendering ──────────────────────────────────────────────────────
    def _render_results(self, names, confs, cals):
        names = list(names)
        confs = list(confs)
        cals = list(cals)
        if len(names) < 3:
            names += ["Unknown"] * (3 - len(names))
            confs += [0.0] * (3 - len(confs))
            cals += [0] * (3 - len(cals))

        # Clear & show results frame
        self.right_placeholder.pack_forget()
        for w in self.results_frame.winfo_children():
            w.destroy()
        self.results_frame.pack(fill="both", expand=True)

        grams = self.serving_var.get()
        macros = estimate_macros(cals[0], grams)

        # ── Top prediction banner ──────────────────────────────────────────────
        banner = tk.Frame(self.results_frame, bg="#1f1205",
                          highlightthickness=1, highlightbackground=ORANGE)
        banner.pack(fill="x", pady=(0,8))
        inner = tk.Frame(banner, bg="#1f1205", padx=16, pady=12)
        inner.pack(fill="x")

        left = tk.Frame(inner, bg="#1f1205")
        left.pack(side="left", fill="both", expand=True)

        tk.Label(left, text="TOP PREDICTION", font=FONT_S,
                 fg="#cbd5e1", bg="#1f1205").pack(anchor="w")
        food_title = names[0].replace("_"," ").title()
        tk.Label(left, text=food_title, font=("Segoe UI",18,"bold"),
                 fg="#f8fafc", bg="#1f1205").pack(anchor="w")
        tk.Label(left, text=f"✓  {confs[0]*100:.1f}% confidence",
                 font=FONT_S, fg=GREEN, bg="#1f1205").pack(anchor="w")

        right = tk.Frame(inner, bg="#1f1205")
        right.pack(side="right")
        tk.Label(right, text=str(macros["Calories"][0]),
                 font=FONT_NUM, fg=ORANGE, bg="#1f1205").pack()
        tk.Label(right, text=f"kcal for {grams}g",
                 font=FONT_S, fg=MUTED, bg="#1f1205").pack()

        # ── Macros grid ────────────────────────────────────────────────────────
        mg = tk.Frame(self.results_frame, bg=BG)
        mg.pack(fill="x", pady=(0,8))

        macro_items = [("Protein", macros["Protein"], GREEN),
                       ("Carbs",   macros["Carbs"],   BLUE),
                       ("Fat",     macros["Fat"],      RED),
                       ("Fiber",   macros["Fiber"],    PURPLE)]

        for i, (label, (val, unit, color), _) in enumerate(macro_items):
            cell = tk.Frame(mg, bg=CARD, highlightthickness=1,
                            highlightbackground=BORDER, width=120)
            cell.grid(row=0, column=i, padx=4, sticky="nsew")
            mg.columnconfigure(i, weight=1)
            tk.Frame(cell, bg=color, height=3).pack(fill="x")
            tk.Label(cell, text=f"{val}{unit}", font=("Segoe UI",13,"bold"),
                     fg=color, bg=CARD, pady=4).pack()
            tk.Label(cell, text=label, font=FONT_S, fg=MUTED, bg=CARD,
                     pady=(0,6)).pack()

        # ── Top 3 bar chart (canvas) ───────────────────────────────────────────
        tk.Label(self.results_frame, text="📊  Top 3 Predictions",
                 font=FONT_H3, fg=TEXT, bg=BG).pack(anchor="w", pady=(4,2))

        chart = tk.Canvas(self.results_frame, bg=CARD, height=110,
                          highlightthickness=1, highlightbackground=BORDER)
        chart.pack(fill="x", pady=(0,8))
        chart.update_idletasks()

        colors = [ORANGE, "#64748b", "#374151"]
        bar_colors_lit = [ORANGE, "#94a3b8", "#475569"]

        def draw_chart(event=None):
            chart.delete("all")
            W = chart.winfo_width()
            pad_l, pad_r, pad_t = 130, 50, 14
            bar_h  = 22
            gap    = 12
            max_w  = W - pad_l - pad_r

            for i, (name, conf, color) in enumerate(zip(names, confs, bar_colors_lit)):
                y = pad_t + i*(bar_h+gap)
                label = name.replace("_"," ").title()[:20]
                chart.create_text(pad_l-6, y+bar_h//2, text=label,
                                  anchor="e", fill=TEXT,
                                  font=("Segoe UI",9))
                bw = max(4, int(conf * max_w))
                chart.create_rectangle(pad_l, y, pad_l+max_w, y+bar_h,
                                       fill="#1a1a1a", outline="")
                chart.create_rectangle(pad_l, y, pad_l+bw, y+bar_h,
                                       fill=color, outline="")
                chart.create_text(pad_l+bw+4, y+bar_h//2,
                                  text=f"{conf*100:.1f}%",
                                  anchor="w", fill=ORANGE,
                                  font=("Segoe UI",8,"bold"))

        chart.bind("<Configure>", draw_chart)
        self.after(50, draw_chart)

        # ── Alternatives ───────────────────────────────────────────────────────
        tk.Label(self.results_frame, text="🔄  Other Possibilities",
                 font=FONT_H3, fg=TEXT, bg=BG).pack(anchor="w", pady=(4,2))

        for i in range(1, 3):
            row = tk.Frame(self.results_frame, bg=CARD,
                           highlightthickness=1, highlightbackground=BORDER)
            row.pack(fill="x", pady=2)
            inner2 = tk.Frame(row, bg=CARD, padx=12, pady=7)
            inner2.pack(fill="x")
            rank = tk.Label(inner2, text=f"#{i+1}", font=FONT_S,
                            fg=ORANGE, bg=CARD)
            rank.pack(side="left", padx=(0,8))
            tk.Label(inner2, text=names[i].replace("_"," ").title(),
                     font=FONT_B, fg=TEXT, bg=CARD).pack(side="left")
            tk.Label(inner2, text=f"{confs[i]*100:.1f}%",
                     font=FONT_S, fg=MUTED, bg=CARD).pack(side="right", padx=(8,0))
            cal_val = int(cals[i] * grams / 100)
            tk.Label(inner2, text=f"{cal_val} kcal",
                     font=("Segoe UI",10,"bold"), fg=ORANGE, bg=CARD).pack(side="right")

        # ── Daily progress bar ─────────────────────────────────────────────────
        daily = 2000
        pct   = min(macros["Calories"][0] / daily, 1.0)
        tk.Label(self.results_frame,
                 text=f"📅  {int(pct*100)}% of daily 2000 kcal goal",
                 font=FONT_S, fg=SUBTEXT, bg=BG).pack(anchor="w", pady=(6,2))

        pb_bg = tk.Canvas(self.results_frame, bg=BORDER, height=10,
                          highlightthickness=0)
        pb_bg.pack(fill="x", pady=(0,16))

        def draw_pb(event=None):
            pb_bg.delete("all")
            W = pb_bg.winfo_width()
            pb_bg.create_rectangle(0,0,W,10,fill=BORDER,outline="")
            pb_bg.create_rectangle(0,0,int(W*pct),10,fill=ORANGE,outline="")

        pb_bg.bind("<Configure>", draw_pb)
        self.after(50, draw_pb)


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = CalorieApp()
    app.mainloop()