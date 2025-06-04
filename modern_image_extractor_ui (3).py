import sys
import os
import csv
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
from matplotlib.widgets import RectangleSelector
from datetime import datetime
import threading
from pathlib import Path
from accuracy_evaluator import AccuracyEvaluator
from processing_timer import ProcessingTimer

class ModernImageToCSVExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("Image to CSV Extractor")
        self.root.geometry("1300x850")
        
        # Initialize performance tracking
        self.timer = ProcessingTimer()
        self.accuracy_evaluator = AccuracyEvaluator()
        
        # Default output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Theme variables
        self.dark_mode = tk.BooleanVar(value=False)
        self.update_theme()

        # Initialize variables
        self.image_paths = []
        self.images = []
        self.images_cv = []
        self.processed_images = []
        self.selected_rows = []
        self.selected_columns = []
        self.current_image_index = 0
        self.selection_mode = "rows"
        self.use_auto_detection = tk.BooleanVar(value=False)
        self.config = {'lang': 'eng', 'config': '--psm 6'}
        self.processing_thread = None

        # Set up the UI
        self.setup_ui()

    def update_theme(self):
        if self.dark_mode.get():
            self.colors = {
                'primary': '#4A90E2',
                'secondary': '#357ABD',
                'accent': '#E94B35',
                'background': '#2C2F33',
                'text': '#FFFFFF',
                'light_text': '#B0B7C0',
                'hover': '#5A9EFF'
            }
        else:
            self.colors = {
                'primary': '#3498db',
                'secondary': '#2980b9',
                'accent': '#e74c3c',
                'background': '#f5f5f5',
                'text': '#333333',
                'light_text': '#7f8c8d',
                'hover': '#5dade2'
            }
        
        self.root.configure(bg=self.colors['background'])
        self.style.configure('TButton', background=self.colors['primary'], foreground='white',
                           font=('Helvetica', 10, 'bold'), padding=8)
        self.style.map('TButton', background=[('active', self.colors['hover'])])
        self.style.configure('Secondary.TButton', background=self.colors['secondary'])
        self.style.map('Secondary.TButton', background=[('active', self.colors['secondary'].replace('35', '2A'))])
        self.style.configure('Accent.TButton', background=self.colors['accent'])
        self.style.map('Accent.TButton', background=[('active', self.colors['accent'].replace('E9', 'C7'))])
        self.style.configure('TFrame', background=self.colors['background'])
        self.style.configure('TLabel', background=self.colors['background'], foreground=self.colors['text'])
        self.style.configure('Header.TLabel', font=('Helvetica', 14, 'bold'), foreground=self.colors['primary'])
        self.style.configure('StatusBar.TLabel', background='#23272A' if self.dark_mode.get() else '#ecf0f1',
                           foreground=self.colors['light_text'])
        self.style.configure('Treeview', background=self.colors['background'], foreground=self.colors['text'],
                           fieldbackground=self.colors['background'])
        self.update_ui_theme()

    def update_ui_theme(self):
        for widget in self.root.winfo_children():
            self._apply_theme_to_widget(widget)

    def _apply_theme_to_widget(self, widget):
        if isinstance(widget, ttk.Frame):
            widget.configure(style='TFrame')
            for child in widget.winfo_children():
                self._apply_theme_to_widget(child)
        elif isinstance(widget, ttk.Label):
            style = 'Header.TLabel' if 'Header' in widget.cget('style') else 'StatusBar.TLabel' if 'StatusBar' in widget.cget('style') else 'TLabel'
            widget.configure(style=style)
        elif isinstance(widget, ttk.Button):
            style = 'TButton'
            if 'Secondary' in widget.cget('style'):
                style = 'Secondary.TButton'
            elif 'Accent' in widget.cget('style'):
                style = 'Accent.TButton'
            widget.configure(style=style)
        elif isinstance(widget, tk.Text):
            widget.configure(bg=self.colors['background'], fg=self.colors['text'])

    def setup_ui(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, pady=(10, 15))
        ttk.Label(header_frame, text="Image to CSV Extractor", style='Header.TLabel').pack(side=tk.LEFT, padx=10)
        ttk.Button(header_frame, text="Toggle Dark Mode", command=self.toggle_theme, style='Secondary.TButton').pack(side=tk.RIGHT, padx=10)

        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        control_canvas = tk.Canvas(main_frame, width=300, bg=self.colors['background'])
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)

        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_frame.bind("<Configure>", lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")))

        self.display_frame = ttk.Frame(main_frame)
        self.display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.setup_file_section(control_frame)
        self.setup_processing_section(control_frame)
        self.setup_selection_section(control_frame)
        self.setup_export_section(control_frame)
        self.setup_display_panel()

        self.status_var = tk.StringVar(value=f"Ready. Load images to begin. {self.get_current_time()}")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, style='StatusBar.TLabel')
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def setup_file_section(self, parent):
        file_frame = ttk.LabelFrame(parent, text="File Management", padding=10)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Button(file_frame, text="Load Images", command=self.load_images).pack(fill=tk.X, pady=2)
        self.image_select_var = tk.StringVar()
        self.image_select_combo = ttk.Combobox(file_frame, textvariable=self.image_select_var, state="readonly")
        self.image_select_combo.pack(fill=tk.X, pady=2)
        self.image_select_combo.bind("<<ComboboxSelected>>", self.switch_image)
        self.filename_var = tk.StringVar(value="No files selected")
        ttk.Label(file_frame, textvariable=self.filename_var, wraplength=280).pack(fill=tk.X, pady=2)

    def setup_processing_section(self, parent):
        process_frame = ttk.LabelFrame(parent, text="Image Processing", padding=10)
        process_frame.pack(fill=tk.X, pady=5)
        type_frame = ttk.Frame(process_frame)
        type_frame.pack(fill=tk.X, pady=2)
        ttk.Label(type_frame, text="Process Type:").pack(side=tk.LEFT)
        self.process_type = tk.StringVar(value="original")
        process_combo = ttk.Combobox(type_frame, textvariable=self.process_type, values=["original", "threshold", "adaptive", "otsu"],
                                   state="readonly", width=12)
        process_combo.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        process_combo.bind("<<ComboboxSelected>>", self.update_processing)

        thresh_frame = ttk.Frame(process_frame)
        thresh_frame.pack(fill=tk.X, pady=2)
        ttk.Label(thresh_frame, text="Threshold:").pack(side=tk.LEFT)
        self.threshold_var = tk.IntVar(value=127)
        threshold_slider = ttk.Scale(thresh_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                                   variable=self.threshold_var, command=self.update_processing)
        threshold_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Label(thresh_frame, textvariable=self.threshold_var, width=3).pack(side=tk.RIGHT, padx=5)

        contrast_frame = ttk.Frame(process_frame)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.contrast_display_var = tk.StringVar(value="1.0")
        self.contrast_var.trace_add("write", lambda *args: self.contrast_display_var.set(f"{self.contrast_var.get():.1f}") or self.update_processing())
        contrast_slider = ttk.Scale(contrast_frame, from_=0.5, to=2.5, orient=tk.HORIZONTAL,
                                  variable=self.contrast_var)
        contrast_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        ttk.Label(contrast_frame, textvariable=self.contrast_display_var, width=3).pack(side=tk.RIGHT, padx=5)
        
        ttk.Button(process_frame, text="Apply Processing", command=self.update_processing).pack(fill=tk.X, pady=2)

    def setup_selection_section(self, parent):
        selection_frame = ttk.LabelFrame(parent, text="Table Selection", padding=10)
        selection_frame.pack(fill=tk.X, pady=5)
        ttk.Checkbutton(selection_frame, text="Automatic Table Detection", variable=self.use_auto_detection,
                       command=self.toggle_auto_detection).pack(anchor=tk.W, pady=2)
        ttk.Separator(selection_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        self.selection_mode_var = tk.StringVar(value="rows")
        ttk.Radiobutton(selection_frame, text="Select Rows", variable=self.selection_mode_var, value="rows",
                       command=self.change_selection_mode).pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(selection_frame, text="Select Columns", variable=self.selection_mode_var, value="columns",
                       command=self.change_selection_mode).pack(anchor=tk.W, pady=2)
        ttk.Separator(selection_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        selection_info = ttk.Frame(selection_frame)
        selection_info.pack(fill=tk.X, pady=2)
        self.rows_count_var = tk.StringVar(value="Rows: 0")
        self.cols_count_var = tk.StringVar(value="Columns: 0")
        ttk.Label(selection_info, textvariable=self.rows_count_var).pack(side=tk.LEFT, padx=5)
        ttk.Label(selection_info, textvariable=self.cols_count_var).pack(side=tk.RIGHT, padx=5)
        ttk.Button(selection_frame, text="Clear Selections", style='Secondary.TButton', command=self.clear_selections).pack(fill=tk.X, pady=2)

    def setup_export_section(self, parent):
        export_frame = ttk.LabelFrame(parent, text="Data Export", padding=10)
        export_frame.pack(fill=tk.X, pady=5)
        ttk.Button(export_frame, text="Extract & Save", command=self.start_background_processing).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Preview Data", command=self.preview_data).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="Help", style='Secondary.TButton', command=self.show_help).pack(fill=tk.X, pady=2)

    def setup_display_panel(self):
        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("No images loaded", color=self.colors['text'])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.patch.set_facecolor(self.colors['background'])
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.display_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.rect_selector = None

    def toggle_theme(self):
        self.dark_mode.set(not self.dark_mode.get())
        self.update_theme()
        self.update_image_display()
        self.status_var.set(f"Theme switched to {'Dark' if self.dark_mode.get() else 'Light'} Mode at {self.get_current_time()}")

    def get_current_time(self):
        return datetime.now().strftime("%I:%M %p WAT, %B %d, %Y")

    def load_images(self):
        start_time = self.timer.start_timer()
        file_paths = filedialog.askopenfilenames(title="Select Image Files", filetypes=(("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All files", "*.*")))
        if file_paths:
            try:
                self.image_paths = list(file_paths)
                self.images = []
                self.images_cv = []
                self.processed_images = []
                self.selected_rows = []
                self.selected_columns = []
                for path in file_paths:
                    image = Image.open(path)
                    image_cv = cv2.imread(path)
                    if image_cv is not None:
                        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                    self.images.append(image)
                    self.images_cv.append(image_cv)
                    self.processed_images.append(image_cv.copy() if image_cv is not None else None)
                    self.selected_rows.append([])
                    self.selected_columns.append([])
                self.current_image_index = 0
                self.update_image_select_combo()
                self.update_image_display()
                self.setup_rectangle_selector()
                self.filename_var.set(f"{len(file_paths)} images loaded")
                self.timer.log_operation("all_images", "load_images", self.timer.end_timer(start_time))
                self.status_var.set(f"Loaded {len(file_paths)} images at {self.get_current_time()}")
                self.update_selection_counts()
                if self.use_auto_detection.get():
                    self.detect_table_structure()
            except Exception as e:
                self.status_var.set(f"Error loading images: {str(e)} at {self.get_current_time()}")
                messagebox.showerror("Error", f"Failed to load images: {str(e)}")

    def update_image_select_combo(self):
        image_names = [os.path.basename(path) for path in self.image_paths]
        self.image_select_combo['values'] = image_names
        if image_names:
            self.image_select_var.set(image_names[0])

    def switch_image(self, event=None):
        if not self.image_paths:
            return
        self.current_image_index = [os.path.basename(path) for path in self.image_paths].index(self.image_select_var.get())
        self.update_image_display()
        self.setup_rectangle_selector()
        self.update_selection_counts()
        self.status_var.set(f"Viewing: {self.image_select_var.get()} at {self.get_current_time()}")
        if self.use_auto_detection.get():
            self.detect_table_structure()

    def update_selection_counts(self):
        if not self.image_paths:
            self.rows_count_var.set("Rows: 0")
            self.cols_count_var.set("Columns: 0")
        else:
            self.rows_count_var.set(f"Rows: {len(self.selected_rows[self.current_image_index])}")
            self.cols_count_var.set(f"Columns: {len(self.selected_columns[self.current_image_index])}")

    def update_processing(self, *args):
        if not self.images_cv or self.current_image_index >= len(self.images_cv) or self.images_cv[self.current_image_index] is None:
            return
        start_time = self.timer.start_timer()
        try:
            img = self.images_cv[self.current_image_index].copy()
            contrast = self.contrast_var.get()
            if contrast != 1.0:
                pil_img = Image.fromarray(img)
                enhancer = ImageEnhance.Contrast(pil_img)
                pil_img = enhancer.enhance(contrast)
                img = np.array(pil_img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            process_type = self.process_type.get()
            if process_type == "original":
                self.processed_images[self.current_image_index] = img
            elif process_type == "threshold":
                _, binary = cv2.threshold(gray, self.threshold_var.get(), 255, cv2.THRESH_BINARY)
                self.processed_images[self.current_image_index] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            elif process_type == "adaptive":
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                self.processed_images[self.current_image_index] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            elif process_type == "otsu":
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.processed_images[self.current_image_index] = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
            self.update_image_display()
            if self.use_auto_detection.get():
                self.detect_table_structure()
            self.timer.log_operation(os.path.basename(self.image_paths[self.current_image_index]), "image_processing", self.timer.end_timer(start_time))
            self.status_var.set(f"Image processed at {self.get_current_time()}")
        except Exception as e:
            self.status_var.set(f"Error processing image: {str(e)} at {self.get_current_time()}")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")

    def update_image_display(self):
        if not self.processed_images or self.current_image_index >= len(self.processed_images) or self.processed_images[self.current_image_index] is None:
            self.ax.clear()
            self.ax.set_title("No images loaded", color=self.colors['text'])
            self.ax.axis('off')
            self.canvas.draw()
            return
        self.ax.clear()
        self.ax.imshow(self.processed_images[self.current_image_index])
        for y_start, y_end in self.selected_rows[self.current_image_index]:
            rect = patches.Rectangle((0, y_start), self.processed_images[self.current_image_index].shape[1], y_end - y_start,
                                   linewidth=1, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
        for x_start, x_end in self.selected_columns[self.current_image_index]:
            rect = patches.Rectangle((x_start, 0), x_end - x_start, self.processed_images[self.current_image_index].shape[0],
                                   linewidth=1, edgecolor='b', facecolor='none')
            self.ax.add_patch(rect)
        self.ax.set_title(f"Image Preview: {os.path.basename(self.image_paths[self.current_image_index])}", color=self.colors['text'])
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()

    def setup_rectangle_selector(self):
        if not self.processed_images or self.current_image_index >= len(self.processed_images) or self.processed_images[self.current_image_index] is None:
            return
        if self.rect_selector:
            self.rect_selector.set_active(False)
        self.rect_selector = RectangleSelector(self.ax, self.on_select_rectangle, useblit=True, button=[1],
                                             minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        self.rect_selector.set_active(not self.use_auto_detection.get())

    def toggle_auto_detection(self):
        if self.use_auto_detection.get():
            self.detect_table_structure()
            self.rect_selector.set_active(False)
            self.status_var.set(f"Automatic table detection enabled at {self.get_current_time()}")
        else:
            self.rect_selector.set_active(True)
            self.status_var.set(f"Automatic table detection disabled at {self.get_current_time()}")

    def detect_table_structure(self):
        if not self.processed_images or self.current_image_index >= len(self.processed_images) or self.processed_images[self.current_image_index] is None:
            return
        start_time = self.timer.start_timer()
        try:
            img = self.processed_images[self.current_image_index].copy()
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.selected_rows[self.current_image_index] = [
                (max(0, cv2.boundingRect(c)[1] - 5), min(img.shape[0], cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] + 5))
                for c in contours if cv2.boundingRect(c)[2] > img.shape[1] * 0.5
            ]
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.selected_columns[self.current_image_index] = [
                (max(0, cv2.boundingRect(c)[0] - 5), min(img.shape[1], cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] + 5))
                for c in contours if cv2.boundingRect(c)[3] > img.shape[0] * 0.5
            ]
            self.selected_rows[self.current_image_index].sort(key=lambda x: x[0])
            self.selected_columns[self.current_image_index].sort(key=lambda x: x[0])
            self.update_image_display()
            self.update_selection_counts()
            self.timer.log_operation(os.path.basename(self.image_paths[self.current_image_index]), "table_detection", self.timer.end_timer(start_time))
            self.status_var.set(f"Table structure detected at {self.get_current_time()}")
        except Exception as e:
            self.status_var.set(f"Error detecting table: {str(e)} at {self.get_current_time()}")
            messagebox.showerror("Error", f"Failed to detect table: {str(e)}")

    def on_select_rectangle(self, eclick, erelease):
        if not self.processed_images or self.current_image_index >= len(self.processed_images) or self.processed_images[self.current_image_index] is None:
            return
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        height, width = self.processed_images[self.current_image_index].shape[:2]
        x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
        y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
        x_start, x_end = min(x1, x2), max(x1, x2)
        y_start, y_end = min(y1, y2), max(y1, y2)
        if self.selection_mode_var.get() == "rows":
            self.selected_rows[self.current_image_index].append((y_start, y_end))
            self.status_var.set(f"Row selection added: {y_start} to {y_end} at {self.get_current_time()}")
        else:
            self.selected_columns[self.current_image_index].append((x_start, x_end))
            self.status_var.set(f"Column selection added: {x_start} to {x_end} at {self.get_current_time()}")
        self.update_image_display()
        self.update_selection_counts()

    def change_selection_mode(self):
        self.selection_mode = self.selection_mode_var.get()
        self.status_var.set(f"Selection mode changed to: {self.selection_mode} at {self.get_current_time()}")

    def clear_selections(self):
        if messagebox.askyesno("Clear Selections", "Are you sure you want to clear all selections for the current image?"):
            if self.current_image_index < len(self.selected_rows):
                self.selected_rows[self.current_image_index] = []
                self.selected_columns[self.current_image_index] = []
                self.update_image_display()
                self.update_selection_counts()
                self.status_var.set(f"All selections cleared at {self.get_current_time()}")

    def extract_data(self):
        if not self.processed_images:
            self.status_var.set(f"No images loaded at {self.get_current_time()}")
            return None
        all_data = []
        try:
            for idx, processed_image in enumerate(self.processed_images):
                if processed_image is None or not self.selected_rows[idx] or not self.selected_columns[idx]:
                    continue
                start_time = self.timer.start_timer()
                self.selected_rows[idx].sort(key=lambda x: x[0])
                self.selected_columns[idx].sort(key=lambda x: x[0])
                table_data = []
                for y_start, y_end in self.selected_rows[idx]:
                    row_data = []
                    for x_start, x_end in self.selected_columns[idx]:
                        cell_img = processed_image[y_start:y_end, x_start:x_end]
                        if cell_img.size == 0:
                            row_data.append("")
                            continue
                        pil_cell_img = Image.fromarray(cell_img)
                        text = pytesseract.image_to_string(pil_cell_img, **self.config).strip()
                        row_data.append(text)
                    table_data.append(row_data)
                all_data.append((os.path.basename(self.image_paths[idx]), table_data))
                self.timer.log_operation(os.path.basename(self.image_paths[idx]), "text_extraction", self.timer.end_timer(start_time))
            self.accuracy_evaluator.evaluate_accuracy(all_data, self.processed_images, self.selected_rows, self.selected_columns)
            return all_data
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error extracting data: {str(e)} at {self.get_current_time()}"))
            return None

    def save_results(self, data):
        if not data:
            return
        try:
            for image_name, table_data in data:
                file_path = self.output_dir / f"{os.path.splitext(image_name)[0]}.csv"
                with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(table_data)
            self.timer.save_timing_to_csv(self.output_dir / "processing_times.csv")
            self.accuracy_evaluator.save_accuracy_to_csv(self.output_dir / "accuracy_metrics.csv")
            self.root.after(0, lambda: self.status_var.set(f"Results saved to {self.output_dir} at {self.get_current_time()}"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Error saving results: {str(e)} at {self.get_current_time()}"))

    def start_background_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.status_var.set(f"Processing already in progress at {self.get_current_time()}")
            return
        self.status_var.set(f"Starting background processing at {self.get_current_time()}")
        self.processing_thread = threading.Thread(target=self.background_process, daemon=True)
        self.processing_thread.start()

    def background_process(self):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Data")
        progress_window.geometry("400x150")
        progress_window.configure(bg=self.colors['background'])
        progress_window.transient(self.root)
        ttk.Label(progress_window, text="Extracting and saving data in background...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', maximum=len(self.processed_images))
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        try:
            data = self.extract_data()
            if data:
                for idx, _ in enumerate(data):
                    progress_bar['value'] = idx + 1
                    progress_window.update()
                self.save_results(data)
            else:
                self.root.after(0, lambda: self.status_var.set(f"No data extracted at {self.get_current_time()}"))
            progress_window.destroy()
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Background processing failed: {str(e)} at {self.get_current_time()}"))
            progress_window.destroy()

    def preview_data(self):
        data = self.extract_data()
        if not data:
            return
        
        preview_window = tk.Toplevel(self.root)
        preview_window.title("Data Preview")
        preview_window.geometry("750x550")
        preview_window.configure(bg=self.colors['background'])
        preview_window.transient(self.root)
        
        preview_frame = ttk.Frame(preview_window, padding=15)
        preview_frame.pack(fill=tk.BOTH, expand=True)
        
        image_select_frame = ttk.Frame(preview_frame)
        image_select_frame.pack(fill=tk.X, pady=10)
        ttk.Label(image_select_frame, text="Select Image:").pack(side=tk.LEFT)
        preview_image_var = tk.StringVar()
        preview_image_combo = ttk.Combobox(image_select_frame, textvariable=preview_image_var, state="readonly")
        preview_image_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        notebook = ttk.Notebook(preview_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        image_names = [image_name for image_name, _ in data]
        preview_image_combo['values'] = image_names
        if image_names:
            preview_image_var.set(image_names[0])
        
        current_tree = [None]
        
        def update_preview(event=None):
            selected_image = preview_image_var.get()
            if not selected_image:
                return
            idx = image_names.index(selected_image)
            for tab in notebook.winfo_children():
                tab.destroy()
            table_frame = ttk.Frame(notebook)
            notebook.add(table_frame, text=selected_image)
            
            table_data = data[idx][1]
            if not table_data or not table_data[0]:
                ttk.Label(table_frame, text="No data extracted for this image.").pack()
                return
                
            cols = [f"Column {i+1}" for i in range(len(table_data[0]))]
            tree = ttk.Treeview(table_frame, columns=cols, show='headings', height=15)
            tree.configure(style='Treeview')
            for i, col in enumerate(cols):
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor='w')
            
            for row_data in table_data:
                tree.insert('', 'end', values=row_data, tags=('editable',))
            
            def on_double_click(event):
                item = tree.identify_row(event.y)
                if item:
                    column = tree.identify_column(event.x)
                    col_idx = int(column.replace('#', '')) - 1
                    if col_idx < len(cols):
                        cell_value = tree.set(item, column)
                        entry = ttk.Entry(table_frame)
                        entry.insert(0, cell_value)
                        entry.place(x=60 + col_idx * 100, y=40 + (tree.index(item) + 1) * 20, width=100)
                        entry.focus_set()
                        def save_edit(event=None):
                            new_value = entry.get().strip()
                            tree.set(item, column, new_value)
                            entry.destroy()
                        entry.bind('<Return>', save_edit)
                        entry.bind('<FocusOut>', save_edit)

            tree.tag_configure('editable', font=('Arial', 10))
            tree.bind('<Double-1>', on_double_click)
            
            v_scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            h_scrollbar = ttk.Scrollbar(table_frame, orient=tk.HORIZONTAL, command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            current_tree[0] = tree
        
        preview_image_combo.bind("<<ComboboxSelected>>", update_preview)
        if image_names:
            update_preview()  
        
        button_frame = ttk.Frame(preview_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(button_frame, text="Save Changes", command=lambda: self.save_and_export(current_tree[0], data, image_names, notebook)).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Close", command=preview_window.destroy).pack(side=tk.RIGHT, padx=5)

    def save_and_export(self, tree, data, image_names, notebook):
        if not tree:
            return
        updated_data = []
        selected_tab = notebook.tab(notebook.select(), "text")
        for image_name, original_data in data:
            if image_name == selected_tab:
                rows = []
                for item in tree.get_children():
                    rows.append(tree.item(item)['values'])
                updated_data.append((image_name, rows))
            else:
                updated_data.append((image_name, original_data))
        self.start_background_processing_with_updated(updated_data)

    def start_background_processing_with_updated(self, data):
        if self.processing_thread and self.processing_thread.is_alive():
            self.status_var.set(f"Processing already in progress at {self.get_current_time()}")
            return
        self.status_var.set(f"Starting background processing of updated data at {self.get_current_time()}")
        self.processing_thread = threading.Thread(target=self.background_process_with_updated, args=(data,), daemon=True)
        self.processing_thread.start()

    def background_process_with_updated(self, data):
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Processing Updated Data")
        progress_window.geometry("400x150")
        progress_window.configure(bg=self.colors['background'])
        progress_window.transient(self.root)
        ttk.Label(progress_window, text="Saving updated data in background...").pack(pady=10)
        progress_bar = ttk.Progressbar(progress_window, mode='determinate', maximum=len(data))
        progress_bar.pack(fill=tk.X, padx=20, pady=5)
        try:
            for idx, _ in enumerate(data):
                progress_bar['value'] = idx + 1
                progress_window.update()
            self.save_results(data)
            progress_window.destroy()
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"Background processing failed: {str(e)} at {self.get_current_time()}"))
            progress_window.destroy()

    def show_help(self):
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("650x550")
        help_window.configure(bg=self.colors['background'])
        help_window.transient(self.root)
        
        help_frame = ttk.Frame(help_window, padding=15)
        help_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(help_frame, text="Image to CSV Extractor Help", style='Header.TLabel').pack(pady=(0, 15))
        help_text = """
How to use the Image to CSV Extractor:

1. Load Images:
   - Click 'Load Images' to select image files (PNG, JPG, etc.).
   - Supports multiple images, displayed in a dropdown.

2. Select an Image:
   - Choose an image from the dropdown to process and view.

3. Process Images:
   - Adjust settings (type: original, threshold, adaptive, otsu; threshold; contrast).
   - Click 'Apply Processing' to update the image view.

4. Define Table Structure:
   - Enable 'Automatic Table Detection' for automatic row/column detection.
   - Or manually select:
     - 'Select Rows' mode: Draw rectangles to define rows.
     - 'Select Columns' mode: Draw rectangles for columns.
     - Draw by clicking and dragging on the image.

5. Extract and Save Data:
   - Click 'Extract & Save' to process images in the background.
   - Data is automatically extracted and saved as CSVs in the 'output' folder.
   - Performance metrics are also saved:
     - 'processing_times.csv': Times for loading, processing, detection, extraction.
     - 'accuracy_metrics.csv': OCR confidence scores (0–1).

6. Preview and Edit Data:
   - Click 'Preview Data' to view extracted tables.
   - Double-click cells to edit.
   - Click 'Save Changes' to save edited data to new CSVs in 'output'.

Tips:
   - Use auto-detection for clear tables to save time.
   - Adjust processing settings for better text visibility.
   - Ensure images have high contrast for best OCR results.
   - Check the 'output' folder for all CSVs.
"""
        text_widget = tk.Text(help_frame, wrap=tk.WORD, height=25, width=75, bg=self.colors['background'], fg=self.colors['text'])
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state='disabled')
        scrollbar = ttk.Scrollbar(help_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(help_frame, text="Close", command=help_window.destroy).pack(pady=10)

if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError:
        print("Error: Tesseract OCR not installed or not in PATH.")
        sys.exit(1)
    
    root = tk.Tk()
    app = ModernImageToCSVExtractor(root)
    root.mainloop()