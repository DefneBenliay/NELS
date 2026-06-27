# NELS
# NELS Software Code
from re import A
import sys
import os
import time
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.widgets import RectangleSelector
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                             QFrame, QScrollArea, QDialog, QSlider, QTableWidget, 
                             QLineEdit, QTableWidgetItem, QMessageBox, QInputDialog, QSplitter,
                             QRadioButton, QButtonGroup, QHeaderView)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# ReportLab Components for PDF Generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table as RLTable, TableStyle as RLTableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# =====================================================================
# 1. THEME & COLORMAP CONFIGURATIONS
# =====================================================================
NESL_THEME = """

QMainWindow, QDialog { background-color: #F5F4F0; }
QWidget { background-color: #F5F4F0; color: #2D2D2D; font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }

/* Panels */
QFrame#Sidebar { background-color: #222E27; border: 1px solid #1C2720; border-radius: 8px; }
QFrame#Sidebar QLabel { color: #E0E5E2; }
QFrame#Sidebar QFrame#WidgetBox { background-color: #222E27; border: 1px solid #3B4D43; }
QFrame#DashboardFrame, QScrollArea#ResultsScroll { background-color: #EBEAE5; border: 1px solid #D1CCC0; border-radius: 8px; }
QFrame#WidgetBox { background-color: #EBEAE5; border: 1px solid #D1CCC0; border-radius: 6px; padding: 6px; }

/* Buttons */
QPushButton {
    background-color: #4A5D4E; color: #FFFFFF; border: none; border-radius: 6px; 
    font-weight: bold; padding: 6px 12px;
}
QPushButton:hover { background-color: #B8A88A; }
QPushButton:pressed { background-color: #8C8984; }
QPushButton:disabled { background-color: #D1CCC0; color: #8C8984; }
QLabel { color: #2D2D2D; }
/* Table */
QTableWidget { background-color: #FFFFFF; color: #2D2D2D; gridline-color: #D1CCC0; border: 1px solid #D1CCC0; border-radius: 4px; }
QTableWidget::item:selected { background-color: #B8A88A; color: #FFFFFF; }
QTableWidget QHeaderView::section { background-color: #EBEAE5; color: #2D2D2D; font-weight: bold; border: 1px solid #D1CCC0; padding: 4px; }
/* Menu Bar */
QMenuBar { background-color: #EBEAE5; color: #2D2D2D; border-bottom: 1px solid #D1CCC0; }
QMenuBar::item:selected { background-color: #B8A88A; color: white; }
QMenu { background-color: #EBEAE5; color: #2D2D2D; border: 1px solid #D1CCC0; }
QMenu::item:selected { background-color: #B8A88A; color: white; }
/* Scrollbar */
QScrollBar:vertical { border: none; background: #F5F4F0; width: 10px; }
QScrollBar::handle:vertical { background: #D1CCC0; border-radius: 5px; }
QScrollBar::handle:vertical:hover { background: #4A5D4E; }
/* Radio Button */
QRadioButton { color: #2D2D2D; font-weight: bold; }
QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 1.5px solid #8C8984; background-color: #F5F4F0; }
QRadioButton::indicator:checked { border: 1.5px solid #4A5D4E; background-color: #4A5D4E; }

"""
COLOR_MAP = {
    "WALKWAY": [255, 255, 255],     
    "SEATING": [255, 192, 203],     
    "PLAYGROUND": [0, 0, 255],      
    "WATER": [0, 255, 255],          
    "SPORTS": [255, 165, 0],         
    "GREEN": [0, 128, 0],            
    "ENTRANCE": [0, 0, 0]            
}

AGENT_COLORS = {
    "Youth": "#FF5733",      
    "Elderly": "#9B59B6",    
    "Parents": "#2ECC71",    
    "Athletes": "#3498DB"    
}

BASE_PREFS = {
    "Youth": {"SEATING": 40.0, "PLAYGROUND": 10.0, "WATER": 50.0, "SPORTS": 100.0, "WALKWAY": 30.0, "GREEN": -999.0, "alpha": 0.1, "beta": 0.2, "count": 100.0},
    "Elderly": {"SEATING": 100.0, "PLAYGROUND": -80.0, "WATER": 60.0, "SPORTS": -90.0, "WALKWAY": 50.0, "GREEN": -999.0, "alpha": 0.3, "beta": 0.5, "count": 100.0},
    "Parents": {"SEATING": 50.0, "PLAYGROUND": 100.0, "WATER": 40.0, "SPORTS": 10.0, "WALKWAY": 30.0, "GREEN": -999.0, "alpha": 0.2, "beta": 0.4, "count": 100.0},
    "Athletes": {"SEATING": -40.0, "PLAYGROUND": -30.0, "WATER": 20.0, "SPORTS": 100.0, "WALKWAY": 80.0, "GREEN": -999.0, "alpha": 0.05, "beta": 0.1, "count": 100.0}
}

AGENT_PREFS = {ag: dict(data) for ag, data in BASE_PREFS.items()}

AGENTS_LIST = ["Youth", "Elderly", "Parents", "Athletes"]
UNITS_LIST = ["SEATING", "PLAYGROUND", "WATER", "SPORTS", "WALKWAY"]

SKALA_RENKLERI = ["#006400", "#90EE90", "#FFFF00", "#FFA500", "#FF0000"]
CMAP_CONTINUOUS_SATISFACTION = LinearSegmentedColormap.from_list("ContinuousSatisfaction", SKALA_RENKLERI)

DIST_MAVI_BEYAZ = ["#00008B", "#4169E1", "#1E90FF", "#87CEFA", "#FFFFFF"]
CMAP_DIST_LEVELS = LinearSegmentedColormap.from_list("DistBlueWhite", DIST_MAVI_BEYAZ, N=5)

CMAPS_AGENTS_DENSITY = {
    "Youth": LinearSegmentedColormap.from_list("Y", ["#FFFFFF", "#FF5733"]),
    "Elderly": LinearSegmentedColormap.from_list("E", ["#FFFFFF", "#9B59B6"]),
    "Parents": LinearSegmentedColormap.from_list("P", ["#FFFFFF", "#2ECC71"]),
    "Athletes": LinearSegmentedColormap.from_list("A", ["#FFFFFF", "#3498DB"])
}

BAR_RENKLERI_CONSENSUS = ["#FF0000", "#FFA500", "#FFFF00", "#90EE90", "#006400"]

# =====================================================================
# 2. CORE MATHEMATICAL MODELING ENGINE
# =====================================================================
class NashLandscapeEvaluator:
    def __init__(self):
        self.w = 150
        self.h = 150
        self.grid = np.zeros((self.h, self.w, 3), dtype=np.uint8) + 128
        self.display_grid = self.grid.copy()
        self.green_mask = np.zeros((self.h, self.w), dtype=bool)
        self.masked_gray_mask = np.zeros((self.h, self.w), dtype=bool)
        self.entrances = []
        self.last_img_path = None
        self.px_per_meter = 1.0

    def load_design_image(self, img, px_per_meter):
        self.last_img_path = "Selected Work Area"
        self.px_per_meter = px_per_meter
        self.h, self.w, _ = img.shape
        self.grid = img.copy()
        self.green_mask = np.zeros((self.h, self.w), dtype=bool)
        self.masked_gray_mask = np.zeros((self.h, self.w), dtype=bool)
        self.display_grid = self.grid.copy()
        
        for y_idx in range(self.h):
            for x_idx in range(self.w):
                area_t = self.get_area_type(self.grid[y_idx, x_idx])
                if area_t == "GREEN":
                    self.green_mask[y_idx, x_idx] = True
                    self.display_grid[y_idx, x_idx] = [210, 210, 210] 
                elif area_t == "MASKED_GRAY":
                    self.masked_gray_mask[y_idx, x_idx] = True
                    self.display_grid[y_idx, x_idx] = [45, 45, 45]
        self.detect_entrances()
        return self.w, self.h

    def detect_entrances(self):
        self.entrances = []
        for y in range(self.h):
            for x in range(self.w):
                if np.all(self.grid[y, x] == COLOR_MAP["ENTRANCE"]):
                    self.entrances.append((x, y))
        if not self.entrances: self.entrances = [(0, 0)]

    def get_area_type(self, rgb):
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        if max(r, g, b) - min(r, g, b) < 15 and r < 240 and r > 30: return "MASKED_GRAY"
        for area, color in COLOR_MAP.items():
            if np.linalg.norm(np.array(rgb) - np.array(color)) < 55: return area
        return "WALKWAY"

    def calculate_all_use_distance_maps(self):
        distance_maps = {}
        for area_name in COLOR_MAP.keys():
            mask = np.zeros((self.h, self.w), dtype=np.uint8)
            for y in range(self.h):
                for x in range(self.w):
                    current_area = self.get_area_type(self.grid[y, x])
                    if current_area == area_name or (area_name == "WALKWAY" and current_area == "ENTRANCE"):
                        mask[y, x] = 1
            if np.sum(mask) == 0: distance_maps[area_name] = np.ones((self.h, self.w)) * 999.0
            else:
                px_distances = cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)
                distance_maps[area_name] = px_distances / self.px_per_meter
        return distance_maps

    def calculate_gate_distance_map(self):
        dist_map = np.zeros((self.h, self.w))
        for y in range(self.h):
            for x in range(self.w):
                area = self.get_area_type(self.grid[y, x])
                if self.green_mask[y, x] or area == "MASKED_GRAY":
                    dist_map[y, x] = np.inf
                    continue
                min_px_dist = min([np.sqrt((x-gx)**2 + (y-gy)**2) for gx, gy in self.entrances])
                dist_map[y, x] = min_px_dist / self.px_per_meter
        return dist_map

    def compute_spatial_density_field(self, placed_agents):
        composite_rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)
        if not placed_agents: return composite_rgb
        max_range = max(self.w, self.h) * 0.45
        for ag in AGENTS_LIST:
            ag_agents = [a for a in placed_agents if a['type'] == ag]
            if not ag_agents: continue
            cohort_surf = np.zeros((self.h, self.w), dtype=np.float32)
            for y in range(self.h):
                for x in range(self.w):
                    dists = np.array([np.sqrt((a['x'] - x)**2 + (a['y'] - y)**2) for a in ag_agents], dtype=np.float32)
                    cohort_surf[y, x] = np.sum(np.exp(-3.0 * dists / max_range))
            if cohort_surf.max() > 0: cohort_surf /= cohort_surf.max()
            composite_rgb += CMAPS_AGENTS_DENSITY[ag](cohort_surf)[:, :, :3]
        if composite_rgb.max() > 0: composite_rgb = (composite_rgb / composite_rgb.max() * 255).astype(np.uint8)
        else: composite_rgb = composite_rgb.astype(np.uint8)
        return composite_rgb

# =====================================================================
# 3. REAL-TIME BACKGROUND SIMULATION SOLVER WORKER
# =====================================================================
class AnalysisWorker(QThread):
    progress_changed = pyqtSignal(int, str)  
    finished_with_result = pyqtSignal(dict, dict, np.ndarray, np.ndarray, np.ndarray)  

    def __init__(self, evaluator, placed_agents):
        super().__init__()
        self.evaluator = evaluator
        self.placed_agents = placed_agents

    def run(self):
        self.progress_changed.emit(5, "Stage 1/4: Compiling Spatial Density Fields...")
        density_rgb = self.evaluator.compute_spatial_density_field(self.placed_agents)
        
        self.progress_changed.emit(25, "Stage 2/4: Interpolating Distance Transform Planes...")
        gate_dist_map = self.evaluator.calculate_gate_distance_map()
        all_use_dist_maps = self.evaluator.calculate_all_use_distance_maps()
        
        agent_utility_maps = {}
        total_population = sum(params["count"] for params in AGENT_PREFS.values())
        if total_population == 0: total_population = 1
        
        lambda_decay = 0.08 

        self.progress_changed.emit(45, "Stage 3/4: Modeling Proximity Matrix Fields...")
        for agent, params in AGENT_PREFS.items():
            u_map = np.zeros((self.evaluator.h, self.evaluator.w))
            
            for y in range(self.evaluator.h):
                for x in range(self.evaluator.w):
                    area = self.evaluator.get_area_type(self.evaluator.grid[y, x])
                    if area == "MASKED_GRAY" or self.evaluator.green_mask[y, x]:
                        u_map[y, x] = -np.inf
                        continue
                        
                    net_utility = 0.0
                    for other_area, weight in params.items():
                        if other_area in ["alpha", "beta", "count", "GREEN"]:
                            continue
                        
                        dist_to_use = all_use_dist_maps[other_area][y, x]
                        if weight == -999.0:
                            if dist_to_use == 0: 
                                net_utility = -np.inf
                                break
                        else:
                            net_utility += weight * np.exp(-lambda_decay * dist_to_use)
                    
                    if net_utility != -np.inf:
                        u_map[y, x] = net_utility - (params["alpha"] * gate_dist_map[y, x])
                    else:
                        u_map[y, x] = -np.inf
                        
            agent_utility_maps[agent] = u_map

        nash_satisfaction = {agent: u_map.copy() for agent, u_map in agent_utility_maps.items()}
        
        total_loops = 5
        for loop in range(total_loops):
            pct = int(50 + (loop / total_loops) * 40)
            self.progress_changed.emit(pct, f"Stage 4/4: Solving Game Theoretical Multi-Cohort Equilibrium ({loop+1}/{total_loops})...")
            
            total_occupancy = np.zeros((self.evaluator.h, self.evaluator.w))
            for agent, params in AGENT_PREFS.items():
                normalized = np.maximum(0, nash_satisfaction[agent])
                if normalized.max() > 0:
                    cohort_weight = (params["count"] / total_population) * 4.0
                    total_occupancy += (normalized / normalized.max()) * cohort_weight

            for agent, params in AGENT_PREFS.items():
                nash_satisfaction[agent] = agent_utility_maps[agent] - (params["beta"] * total_occupancy)
                nash_satisfaction[agent][agent_utility_maps[agent] == -np.inf] = -np.inf

        collective_map = np.zeros((self.evaluator.h, self.evaluator.w))
        total_weight_grid = np.zeros((self.evaluator.h, self.evaluator.w))
        
        for agent, params in AGENT_PREFS.items():
            valid_mask = nash_satisfaction[agent] != -np.inf
            weight = params["count"]
            collective_map[valid_mask] += nash_satisfaction[agent][valid_mask] * weight
            total_weight_grid[valid_mask] += weight
            
        collective_map = np.divide(collective_map, total_weight_grid, out=np.zeros_like(collective_map), where=total_weight_grid!=0)
        
        self.progress_changed.emit(100, "Simulation process finished successfully!")
        self.finished_with_result.emit(agent_utility_maps, nash_satisfaction, collective_map, gate_dist_map, density_rgb)


# =====================================================================
# 3.5 AREA SELECTION & CROPPING DIALOG WINDOW
# =====================================================================
class AreaSelectionDialog(QDialog):
    def __init__(self, img, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Area of Interest (Crop)")
        self.setStyleSheet(NESL_THEME)
        self.setMinimumSize(850, 700)
        self.img = img
        self.crop_coords = None  # (x1, y1, x2, y2)
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        info_label = QLabel("Drag a box with the left mouse button to crop the workspace.\n"
                            "Click 'Confirm Selection' to crop, or 'Use Full Image' to keep the original size.")
        info_label.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        info_label.setStyleSheet(NESL_THEME)
        layout.addWidget(info_label)
        
        self.figure, self.ax = plt.subplots(figsize=(6, 5))
        self.figure.patch.set_facecolor("#EBEAE5")
        self.ax.set_facecolor("#FFFFFF")
        self.ax.imshow(self.img)
        self.ax.axis('on')
        self.ax.tick_params(colors='#2D2D2D', labelsize=8)
        self.figure.tight_layout()
        
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.selector = RectangleSelector(
            self.ax, self.on_select,
            useblit=True,
            button=[1],  # Left button
            interactive=True,
            props=dict(facecolor='#4A5D4E', edgecolor='#ffffff', alpha=0.3, fill=True)
        )
        
        btn_layout = QHBoxLayout()
        self.btn_confirm = QPushButton("Confirm Selection")
        self.btn_confirm.setMinimumHeight(35)
        self.btn_confirm.setStyleSheet(NESL_THEME)
        self.btn_confirm.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_confirm)
        
        self.btn_save = QPushButton("💾 Save")
        self.btn_save.setMinimumHeight(35)
        self.btn_save.setStyleSheet(NESL_THEME)
        self.btn_save.clicked.connect(self.save_crop_coords)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_load = QPushButton("📂 Load")
        self.btn_load.setMinimumHeight(35)
        self.btn_load.setStyleSheet(NESL_THEME)
        self.btn_load.clicked.connect(self.load_crop_coords)
        btn_layout.addWidget(self.btn_load)
        
        self.btn_skip = QPushButton("Use Full Image (Skip)")
        self.btn_skip.setMinimumHeight(35)
        self.btn_skip.setStyleSheet(NESL_THEME)
        self.btn_skip.clicked.connect(self.skip_crop)
        btn_layout.addWidget(self.btn_skip)
        
        layout.addLayout(btn_layout)
        
    def on_select(self, eclick, erelease):
        x1, y1 = int(round(eclick.xdata)), int(round(eclick.ydata))
        x2, y2 = int(round(erelease.xdata)), int(round(erelease.ydata))
        
        h, w, _ = self.img.shape
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        
        self.crop_coords = (x_min, y_min, x_max, y_max)
        
    def skip_crop(self):
        self.crop_coords = None
        self.reject()

    def save_crop_coords(self):
        if self.crop_coords is None:
            QMessageBox.warning(self, "No Selection", "Please drag a box to select an area first!")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Crop Coordinates", "crop_coordinates.json", "JSON Files (*.json)")
        if save_path:
            import json
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "x_min": self.crop_coords[0],
                        "y_min": self.crop_coords[1],
                        "x_max": self.crop_coords[2],
                        "y_max": self.crop_coords[3]
                    }, f, indent=4)
                QMessageBox.information(self, "Success", "Crop coordinates saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save coordinates:\n{str(e)}")

    def load_crop_coords(self):
        load_path, _ = QFileDialog.getOpenFileName(self, "Load Crop Coordinates", "", "JSON Files (*.json)")
        if load_path:
            import json
            try:
                with open(load_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                x_min = int(data["x_min"])
                y_min = int(data["y_min"])
                x_max = int(data["x_max"])
                y_max = int(data["y_max"])
                
                # Check dimensions against current image limits
                h, w, _ = self.img.shape
                x_min = max(0, min(x_min, w - 1))
                x_max = max(0, min(x_max, w - 1))
                y_min = max(0, min(y_min, h - 1))
                y_max = max(0, min(y_max, h - 1))
                
                self.crop_coords = (x_min, y_min, x_max, y_max)
                
                # Update RectangleSelector visualization on screen
                self.selector.extents = (x_min, x_max, y_min, y_max)
                self.canvas.draw_idle()
                
                QMessageBox.information(self, "Success", "Crop coordinates loaded and applied successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load coordinates:\n{str(e)}")


# =====================================================================
# 4. MUTABLE GLOBAL PARAMETER MATRIX DIALOGUE WINDOW
# =====================================================================
class ParameterDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Cohort Preference Matrices")
        self.setStyleSheet(NESL_THEME)
        self.setMinimumWidth(820)
        self.setMinimumHeight(450)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout(self)
        
        info_title = QLabel("Integrated Attraction / Repulsion Preference Multi-Matrix Editor")
        info_title.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        info_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(info_title)

        self.table = QTableWidget()
        self.table.setRowCount(4)
        self.table.setColumnCount(8) 
        self.table.setVerticalHeaderLabels(AGENTS_LIST)
        self.table.setHorizontalHeaderLabels(["SEATING", "PLAYGROUND", "WATER", "SPORTS", "WALKWAY", "ALPHA (α)", "BETA (β)", "COUNT (ppl)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        
        self.populate_fields()
        layout.addWidget(self.table)

        btn_save = QPushButton("Save Configurations Matrix & Close")
        btn_save.setMinimumHeight(40)
        btn_save.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        btn_save.setStyleSheet(NESL_THEME)
        btn_save.clicked.connect(self.save_and_accept)
        layout.addWidget(btn_save)

    def populate_fields(self):
        for r_idx, agent in enumerate(AGENTS_LIST):
            for c_idx, area in enumerate(UNITS_LIST):
                val = str(AGENT_PREFS[agent][area])
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(r_idx, c_idx, item)
            
            item_a = QTableWidgetItem(str(AGENT_PREFS[agent]["alpha"]))
            item_a.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r_idx, 5, item_a)
            
            item_b = QTableWidgetItem(str(AGENT_PREFS[agent]["beta"]))
            item_b.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r_idx, 6, item_b)

            item_c = QTableWidgetItem(str(int(AGENT_PREFS[agent]["count"])))
            item_c.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.table.setItem(r_idx, 7, item_c)

    def save_and_accept(self):
        try:
            for r_idx, agent in enumerate(AGENTS_LIST):
                for c_idx, area in enumerate(UNITS_LIST):
                    AGENT_PREFS[agent][area] = float(self.table.item(r_idx, c_idx).text())
                AGENT_PREFS[agent]["alpha"] = float(self.table.item(r_idx, 5).text())
                AGENT_PREFS[agent]["beta"] = float(self.table.item(r_idx, 6).text())
                AGENT_PREFS[agent]["count"] = float(self.table.item(r_idx, 7).text())
            self.accept()
        except ValueError:
            QMessageBox.critical(self, "Validation Error", "Ensure all structural matrix fields contain numeric parameters.")


# =====================================================================
# 4.5 PAPER ADD-ON DIALOG WINDOW
# =====================================================================
class PaperAddonDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Paper Add-on Preview")
        self.setStyleSheet(NESL_THEME)
        self.resize(550, 680)
        self.main_window = main_window
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)
        
        info = QLabel("300 DPI Resolution Paper Export Layout")
        info.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        info.setStyleSheet(NESL_THEME)
        layout.addWidget(info)
        
        # 1. Academic Standards & Aspect Ratio
        default_w_cm = 15.0
        default_h_cm = 31.0
        
        # Size Controls Layout
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 5, 0, 5)
        
        control_layout.addWidget(QLabel("Width (cm):"))
        self.txt_width = QLineEdit(f"{default_w_cm:.2f}")
        self.txt_width.setFixedWidth(60)
        self.txt_width.setStyleSheet(NESL_THEME)
        control_layout.addWidget(self.txt_width)
        
        control_layout.addWidget(QLabel("Height (cm):"))
        self.txt_height = QLineEdit(f"{default_h_cm:.2f}")
        self.txt_height.setFixedWidth(60)
        self.txt_height.setStyleSheet(NESL_THEME)
        control_layout.addWidget(self.txt_height)
        
        self.btn_update = QPushButton("🔄 Update Preview")
        self.btn_update.setStyleSheet(NESL_THEME)
        self.btn_update.clicked.connect(self.apply_new_size)
        control_layout.addWidget(self.btn_update)
        
        control_layout.addStretch()
        
        self.lbl_pixels = QLabel()
        self.lbl_pixels.setStyleSheet(NESL_THEME)
        self.update_pixel_label(default_w_cm, default_h_cm)
        control_layout.addWidget(self.lbl_pixels)
        
        layout.addLayout(control_layout)
        
        # Figure initialization
        dpi_target = 100 # Preview resolution
        self.figure = plt.figure(figsize=(default_w_cm / 2.54, default_h_cm / 2.54), dpi=dpi_target, facecolor='#FFFFFF')
        
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Act as if the update button has been clicked on start
        self.apply_new_size()
        
        # Buttons Layout
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("💾 Save Image (PNG)")
        self.btn_save.setMinimumHeight(35)
        self.btn_save.setStyleSheet(NESL_THEME)
        self.btn_save.clicked.connect(self.save_image)
        btn_layout.addWidget(self.btn_save)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.setMinimumHeight(35)
        self.btn_close.setStyleSheet(NESL_THEME)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)
        
        layout.addLayout(btn_layout)

    def draw_layout(self, w_cm, h_cm):
        self.figure.clear()
        
        # Set layout engine to constrained to align axes edges perfectly
        self.figure.set_layout_engine(layout='constrained')
        self.figure.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0.02, wspace=0.02)
        
        inch_width = w_cm / 2.54
        inch_height = h_cm / 2.54
        self.figure.set_size_inches(inch_width, inch_height, forward=True)
        
        gs_pub = self.figure.add_gridspec(2, 4, height_ratios=[1, 4])
        
        w_meters = self.main_window.evaluator.w / self.main_window.evaluator.px_per_meter
        h_meters = self.main_window.evaluator.h / self.main_window.evaluator.px_per_meter
        spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
        
        overlay_mask = np.zeros((self.main_window.evaluator.h, self.main_window.evaluator.w, 4))
        overlay_mask[self.main_window.evaluator.green_mask, :3] = 0.33
        overlay_mask[self.main_window.evaluator.green_mask, 3] = 1.0
        overlay_mask[self.main_window.evaluator.masked_gray_mask, :3] = 0.83
        overlay_mask[self.main_window.evaluator.masked_gray_mask, 3] = 1.0
        
        for col_idx, ag in enumerate(AGENTS_LIST):
            ax_sub = self.figure.add_subplot(gs_pub[0, col_idx])
            ax_sub.set_facecolor("#FFFFFF")
            
            data_masked = self.main_window.current_report_data[ag]["matrix"]
            ax_sub.imshow(data_masked, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
            ax_sub.imshow(overlay_mask, extent=[0, w_meters, h_meters, 0])
            
            ax_sub.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
            ax_sub.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
            ax_sub.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.4, alpha=1.0)
            ax_sub.tick_params(colors='#2D2D2D', labelsize=4, pad=2)
            for spine in ax_sub.spines.values():
                spine.set_color('#D1CCC0')
                
            ax_sub.set_title(f"{ag.upper()} PREFERENCE", fontsize=5, fontweight='bold', color='#111111')
            
        ax_bottom = self.figure.add_subplot(gs_pub[1, :])
        ax_bottom.set_facecolor("#FFFFFF")
        
        collective_matrix = self.main_window.current_report_data["Collective"]["matrix"]
        ax_bottom.imshow(collective_matrix, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
        ax_bottom.imshow(overlay_mask, extent=[0, w_meters, h_meters, 0])
        
        ax_bottom.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
        ax_bottom.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
        ax_bottom.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.4, alpha=1.0)
        ax_bottom.tick_params(colors='#2D2D2D', labelsize=5, pad=3)
        for spine in ax_bottom.spines.values():
            spine.set_color('#D1CCC0')
            
        ax_bottom.set_title("Nash Equilibrium Weighted Collective Consensus Layer", fontsize=8, fontweight='bold', color='#111111', pad=6)
        
        self.canvas.draw()

    def apply_new_size(self):
        try:
            w_cm = float(self.txt_width.text())
            h_cm = float(self.txt_height.text())
            if w_cm <= 0 or h_cm <= 0:
                raise ValueError("Dimensions must be positive values.")
            
            self.draw_layout(w_cm, h_cm)
            self.update_pixel_label(w_cm, h_cm)
        except Exception as e:
            QMessageBox.critical(self, "Invalid Input", f"Please enter valid numeric dimensions:\n{str(e)}")

    def update_pixel_label(self, w_cm, h_cm):
        w_px = int((w_cm / 2.54) * 300)
        h_px = int((h_cm / 2.54) * 300)
        self.lbl_pixels.setText(f"ℹ️ Output Resolution: {w_px} x {h_px} px (at 300 DPI)")

    def save_image(self):
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Paper Layout Image", "Paper_Addon_Layout.png", "Images (*.png)")
        if save_path:
            try:
                self.figure.savefig(save_path, dpi=300, facecolor=self.figure.get_facecolor(), bbox_inches='tight')
                QMessageBox.information(self, "Success", "Layout image saved successfully at 300 DPI!")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save image:\n{str(e)}")

    def closeEvent(self, event):
        plt.close(self.figure)
        super().closeEvent(event)

# =====================================================================
# 4.6 NODE PLACEMENT CONFIGURATOR DIALOG
# =====================================================================
class NodePlacementConfiguratorDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Node Placement Configurator")
        self.setStyleSheet(NESL_THEME)
        self.resize(1350, 850)
        
        # State variables for zooming/panning (default to full size)
        w_meters = self.main_window.evaluator.w / self.main_window.evaluator.px_per_meter
        h_meters = self.main_window.evaluator.h / self.main_window.evaluator.px_per_meter
        self.cur_xlim = (0, w_meters)
        self.cur_ylim = (h_meters, 0)
        self.is_panning = False
        self.pan_start_x = None
        self.pan_start_y = None
        self.transfer_pressed = False
        
        self.initUI()
        
    def initUI(self):
        layout = QHBoxLayout(self)
        
        # Splitter to separate map canvas and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # Left Panel: Main Map Canvas (WORKSPACE MATRIX CORE LAYOUT)
        map_frame = QFrame()
        map_frame.setObjectName("DashboardFrame")
        map_layout = QVBoxLayout(map_frame)
        
        self.figure = plt.figure(figsize=(9, 8), dpi=120)
        self.figure.patch.set_facecolor("#EBEAE5")
        self.ax_main = self.figure.add_subplot(111)
        self.ax_main.set_facecolor("#FFFFFF")
        
        self.canvas = FigureCanvas(self.figure)
        
        # Connect Matplotlib events for node placement and pan/zoom
        self.canvas.mpl_connect('button_press_event', self.on_map_click)
        self.canvas.mpl_connect('button_release_event', self.on_map_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_map_pan_move)
        self.canvas.mpl_connect('scroll_event', self.on_map_scroll)
        
        map_layout.addWidget(self.canvas)
        splitter.addWidget(map_frame)
        
        # Right Panel: Agent Select, Density Map Preview, and Transfer buttons
        right_panel = QFrame()
        right_panel.setObjectName("DashboardFrame")
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(10, 12, 10, 12)
        right_panel_layout.setSpacing(10)
        
        right_panel_layout.addWidget(QLabel("<h3>NODE PLACEMENT CONFIGURATOR</h3>"))
        
        # Radio button selection
        self.agent_group = QButtonGroup(self)
        self.counter_labels = {}
        
        for ag in AGENTS_LIST:
            row_layout = QHBoxLayout()
            count_lbl = QLabel("<b>[ 0 ]</b>")
            count_lbl.setFixedWidth(45)
            count_lbl.setStyleSheet(f"color: {AGENT_COLORS[ag]}; font-size: 13px; font-weight: bold;")
            self.counter_labels[ag] = count_lbl
            row_layout.addWidget(count_lbl)
            
            rbtn = QRadioButton(f"{ag} Cohort Target")
            if ag == self.main_window.selected_agent_type: 
                rbtn.setChecked(True)
            rbtn.setProperty("agent_type", ag)
            rbtn.setStyleSheet(f"QRadioButton {{ color: #303030; font-weight: bold; }} QRadioButton::indicator:checked {{ background-color: {AGENT_COLORS[ag]}; }}")
            self.agent_group.addButton(rbtn)
            row_layout.addWidget(rbtn)
            right_panel_layout.addLayout(row_layout)
            
        self.agent_group.buttonToggled.connect(self.on_agent_selection_changed)
        
        # Add Project Repository Excel tools (Carried on Configurator)
        right_panel_layout.addSpacing(5)
        lbl_excel_title = QLabel("<b>PROJECT REPOSITORY</b>")
        lbl_excel_title.setStyleSheet("font-weight: bold; color: #2D2D2D;")
        right_panel_layout.addWidget(lbl_excel_title)
        
        self.btn_import_excel = QPushButton("📥 IMPORT COMPLETE PROJECT DATA")
        self.btn_import_excel.setStyleSheet(NESL_THEME)
        self.btn_import_excel.clicked.connect(self.handle_import_excel)
        right_panel_layout.addWidget(self.btn_import_excel)

        self.btn_export_excel = QPushButton("📤 EXPORT COMPLETE PROJECT DATA")
        self.btn_export_excel.setStyleSheet(NESL_THEME)
        
        self.btn_export_excel.clicked.connect(self.handle_export_excel)
        right_panel_layout.addWidget(self.btn_export_excel)

        self.btn_density_preview = QPushButton("📈 DENSITY MAPPING PREVIEW")
        self.btn_density_preview.setStyleSheet(NESL_THEME)
        self.btn_density_preview.clicked.connect(self.preview_density_mapping)
        right_panel_layout.addWidget(self.btn_density_preview)
        
        right_panel_layout.addSpacing(5)
        
        # Geospatial Density Mapping Preview Panel
        right_panel_layout.addWidget(QLabel("<b>Geospatial Density Mapping Preview</b>"))
        self.density_preview_figure, self.density_preview_ax = plt.subplots(figsize=(4.0, 4.0))
        self.density_preview_figure.patch.set_facecolor("#EBEAE5")
        self.density_preview_canvas = FigureCanvas(self.density_preview_figure)
        self.density_preview_ax.axis('off')
        self.density_preview_figure.tight_layout()
        right_panel_layout.addWidget(self.density_preview_canvas)
        
        right_panel_layout.addStretch()
        
        # En altta: TRANSFER TO PREFERENCE MATRIX button
        self.btn_transfer_matrix = QPushButton("🔄 TRANSFER TO PREFERENCE MATRIX")
        self.btn_transfer_matrix.setMinimumHeight(40)
        self.btn_transfer_matrix.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        self.btn_transfer_matrix.setStyleSheet((NESL_THEME))
        self.btn_transfer_matrix.clicked.connect(self.transfer_live_counts_to_prefs_matrix)
        right_panel_layout.addWidget(self.btn_transfer_matrix)
        
        self.btn_close = QPushButton("Close Configurator")
        self.btn_close.setMinimumHeight(35)
        self.btn_close.setStyleSheet("background-color: #8C8984; color: #FFFFFF; border: none; border-radius: 6px; font-weight: bold;")
        self.btn_close.clicked.connect(self.accept)
        right_panel_layout.addWidget(self.btn_close)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([900, 380])
        
        # Draw the initial map layout and counts
        self.sync_and_recalculate_counts()
        self.render_placement_plots()
        self.render_density_preview()
        
    def render_density_preview(self):
        self.density_preview_ax.clear()
        if self.main_window.latest_density_rgb is not None:
            w_meters = self.main_window.evaluator.w / self.main_window.evaluator.px_per_meter
            h_meters = self.main_window.evaluator.h / self.main_window.evaluator.px_per_meter
            self.density_preview_ax.imshow(self.main_window.latest_density_rgb, extent=[0, w_meters, h_meters, 0])
            
            overlay_preview = np.zeros((self.main_window.evaluator.h, self.main_window.evaluator.w, 4))
            overlay_preview[self.main_window.evaluator.green_mask, :3] = 0.33
            overlay_preview[self.main_window.evaluator.green_mask, 3] = 1.0
            overlay_preview[self.main_window.evaluator.masked_gray_mask, :3] = 0.83
            overlay_preview[self.main_window.evaluator.masked_gray_mask, 3] = 1.0
            self.density_preview_ax.imshow(overlay_preview, extent=[0, w_meters, h_meters, 0])
            
            spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
            self.density_preview_ax.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
            self.density_preview_ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
            self.density_preview_ax.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.5, alpha=1.0)
            self.density_preview_ax.tick_params(colors='#2D2D2D', labelsize=6)
            self.density_preview_ax.axis('on')
        else:
            self.density_preview_ax.text(0.5, 0.5, "Click 'DENSITY MAPPING PREVIEW' to render...", color='gray', ha='center', va='center')
            self.density_preview_ax.axis('off')
        self.density_preview_figure.tight_layout()
        self.density_preview_canvas.draw()
        
    def preview_density_mapping(self):
        density_rgb = self.main_window.evaluator.compute_spatial_density_field(self.main_window.placed_agents)
        self.density_preview_ax.clear()
        w_meters = self.main_window.evaluator.w / self.main_window.evaluator.px_per_meter
        h_meters = self.main_window.evaluator.h / self.main_window.evaluator.px_per_meter
        self.density_preview_ax.imshow(density_rgb, extent=[0, w_meters, h_meters, 0])
        
        overlay_preview = np.zeros((self.main_window.evaluator.h, self.main_window.evaluator.w, 4))
        overlay_preview[self.main_window.evaluator.green_mask, :3] = 0.33
        overlay_preview[self.main_window.evaluator.green_mask, 3] = 1.0
        overlay_preview[self.main_window.evaluator.masked_gray_mask, :3] = 0.83
        overlay_preview[self.main_window.evaluator.masked_gray_mask, 3] = 1.0
        self.density_preview_ax.imshow(overlay_preview, extent=[0, w_meters, h_meters, 0])
        
        spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
        self.density_preview_ax.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
        self.density_preview_ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
        self.density_preview_ax.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.5, alpha=1.0)
        self.density_preview_ax.tick_params(colors='#2D2D2D', labelsize=6)
        self.density_preview_ax.axis('on')
        self.density_preview_figure.tight_layout()
        self.density_preview_canvas.draw()
        
    def on_map_scroll(self, event):
        if event.inaxes != self.ax_main: return
        base_scale = 1.2
        cur_x_min, cur_x_max = self.ax_main.get_xlim()
        cur_y_min, cur_y_max = self.ax_main.get_ylim()
        xdata = event.xdata
        ydata = event.ydata
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1.0
        self.cur_xlim = (xdata - (xdata - cur_x_min) * scale_factor, xdata + (cur_x_max - xdata) * scale_factor)
        self.cur_ylim = (ydata - (ydata - cur_y_min) * scale_factor, ydata + (cur_y_max - ydata) * scale_factor)
        self.ax_main.set_xlim(self.cur_xlim)
        self.ax_main.set_ylim(self.cur_ylim)
        self.canvas.draw()
        
    def on_map_click(self, event):
        if event.inaxes != self.ax_main: return
        if event.button == 2:
            self.is_panning = True
            self.pan_start_x = event.xdata
            self.pan_start_y = event.ydata
            return
        if event.button == 1:
            px = event.xdata * self.main_window.evaluator.px_per_meter
            py = event.ydata * self.main_window.evaluator.px_per_meter
            x, y = int(round(px)), int(round(py))
            if 0 <= x < self.main_window.evaluator.w and 0 <= y < self.main_window.evaluator.h:
                if self.main_window.evaluator.green_mask[y, x] or self.main_window.evaluator.masked_gray_mask[y, x]:
                    return
                self.main_window.placed_agents.append({
                    'type': self.main_window.selected_agent_type,
                    'x': x,
                    'y': y
                })
                self.sync_and_recalculate_counts()
                self.render_placement_plots()
        elif event.button == 3:
            px = event.xdata * self.main_window.evaluator.px_per_meter
            py = event.ydata * self.main_window.evaluator.px_per_meter
            if len(self.main_window.placed_agents) > 0:
                dists = [np.hypot(a['x'] - px, a['y'] - py) for a in self.main_window.placed_agents]
                min_idx = np.argmin(dists)
                if dists[min_idx] < 30.0:
                    self.main_window.placed_agents.pop(min_idx)
                    self.sync_and_recalculate_counts()
                    self.render_placement_plots()
                    
    def on_map_pan_move(self, event):
        if not self.is_panning or event.inaxes != self.ax_main: return
        dx = event.xdata - self.pan_start_x
        dy = event.ydata - self.pan_start_y
        cur_x_min, cur_x_max = self.ax_main.get_xlim()
        cur_y_min, cur_y_max = self.ax_main.get_ylim()
        self.cur_xlim = (cur_x_min - dx, cur_x_max - dx)
        self.cur_ylim = (cur_y_min - dy, cur_y_max - dy)
        self.ax_main.set_xlim(self.cur_xlim)
        self.ax_main.set_ylim(self.cur_ylim)
        self.canvas.draw()
        
    def on_map_release(self, event):
        if event.button == 2: self.is_panning = False
        
    def on_agent_selection_changed(self, button, checked):
        if checked: 
            self.main_window.selected_agent_type = button.property("agent_type")
            
    def handle_import_excel(self):
        self.main_window.import_complete_project_from_excel()
        self.sync_and_recalculate_counts()
        self.render_placement_plots()
        self.render_density_preview()
        
    def handle_export_excel(self):
        self.main_window.export_complete_project_to_excel()
            
    def sync_and_recalculate_counts(self):
        counts = {ag: 0 for ag in AGENTS_LIST}
        for agent in self.main_window.placed_agents: 
            counts[agent['type']] += 1
        for ag in AGENTS_LIST:
            self.counter_labels[ag].setText(f"<b>[ {counts[ag]} ]</b>")
        self.main_window.update_live_matrix_table()
        self.main_window.invalidate_simulation_matrices()
        
    def transfer_live_counts_to_prefs_matrix(self):
        total_agents = len(self.main_window.placed_agents)
        if total_agents == 0: return
        for idx, ag in enumerate(AGENTS_LIST):
            live_count = sum(1 for a in self.main_window.placed_agents if a['type'] == ag)
            p_ratio = live_count / total_agents
            for unit in UNITS_LIST:
                AGENT_PREFS[ag][unit] = float(BASE_PREFS[ag][unit] * p_ratio)
            AGENT_PREFS[ag]["alpha"] = float(BASE_PREFS[ag]["alpha"] * p_ratio)
            AGENT_PREFS[ag]["beta"] = float(BASE_PREFS[ag]["beta"] * p_ratio)
            AGENT_PREFS[ag]["count"] = float(live_count)
        self.main_window.update_live_matrix_table()
        self.main_window.invalidate_simulation_matrices()
        self.transfer_pressed = True
        self.main_window.lbl_status.setText("✅ Attraction and counts transferred into global preference matrix.")
        QMessageBox.information(self, "Transfer Completed", "Live calculated attraction matrix and agent counts synchronized.")
        
    def render_placement_plots(self):
        self.ax_main.clear()
        self.ax_main.set_facecolor('#FFFFFF')
        w_meters = self.main_window.evaluator.w / self.main_window.evaluator.px_per_meter
        h_meters = self.main_window.evaluator.h / self.main_window.evaluator.px_per_meter
        if self.main_window.map_loaded:
            self.ax_main.imshow(self.main_window.evaluator.display_grid, extent=[0, w_meters, h_meters, 0])
            for agent in self.main_window.placed_agents:
                self.ax_main.scatter(agent['x'] / self.main_window.evaluator.px_per_meter,
                                     agent['y'] / self.main_window.evaluator.px_per_meter,
                                     color=AGENT_COLORS[agent['type']], edgecolors='white', s=55, zorder=5)
            self.ax_main.set_xlim(self.cur_xlim)
            self.ax_main.set_ylim(self.cur_ylim)
            spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
            self.ax_main.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
            self.ax_main.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
            self.ax_main.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.5, alpha=1.0)
            self.ax_main.tick_params(colors='#2D2D2D', labelsize=6)
            for spine in self.ax_main.spines.values(): spine.set_color('#D1CCC0')
            self.ax_main.set_title("WORKSPACE MATRIX CORE LAYOUT", fontsize=8, fontweight='bold', color="#4A5D4E")
        else:
            self.ax_main.text(0.5, 0.5, "Upload layout backdrop template image in main window first...", color='gray', ha='center', va='center')
            self.ax_main.axis('off')
        self.canvas.draw_idle()

    def closeEvent(self, event):
        plt.close(self.figure)
        plt.close(self.density_preview_figure)
        super().closeEvent(event)

# =====================================================================
# 5. MAIN SYSTEM DASHBOARD INTERFACE
# =====================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nash Equilibrium Landscape Multi-Distance Simulator Dashboard")
        self.resize(1850, 980)
        self.evaluator = NashLandscapeEvaluator()
        
        self.placed_agents = [] 
        self.selected_agent_type = "Youth"
        self.map_loaded = False 
        self.nash_analyzed = False 
        
        self.current_report_data = {}
        self.latest_nash_matrix = None 
        self.latest_density_rgb = None 
        self.latest_nash_maps = {}

        self.setStyleSheet(NESL_THEME)
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------- LEFT SIDEBAR PANEL ----------------
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(10, 12, 10, 12)
        sidebar_layout.setSpacing(10)
        
        # Design layout blueprint canvas at the top
        self.input_figure, self.input_ax = plt.subplots(figsize=(4.5, 4.5))
        self.input_figure.patch.set_facecolor("#222E27")
        self.input_canvas = FigureCanvas(self.input_figure)
        self.input_ax.axis('off')
        self.input_figure.tight_layout()
        sidebar_layout.addWidget(self.input_canvas)

        self.btn_load = QPushButton("UPLOAD DESIGN LAYOUT (.PNG)")
        self.btn_load.setMinimumHeight(40)
        self.btn_load.clicked.connect(self.load_image_and_analyze)
        sidebar_layout.addWidget(self.btn_load)
        
        self.lbl_dimensions = QLabel("<b>Map Status:</b> Waiting for workspace template...")
        self.lbl_dimensions.setStyleSheet("background-color: transparent; color: #E74C3C; font-size: 11px;")
        sidebar_layout.addWidget(self.lbl_dimensions)

        self.btn_menu = QPushButton("EDIT COHORT PREFERENCE MATRICES")
        self.btn_menu.setMinimumHeight(35)
        self.btn_menu.setEnabled(False)
        self.btn_menu.clicked.connect(self.open_parameter_menu)
        sidebar_layout.addWidget(self.btn_menu)
        
        sidebar_layout.addSpacing(5)

        # Node Placement Configurator button (Requirement 2)
        self.btn_node_config = QPushButton("📍 NODE PLACEMENT CONFIGURATOR")
        self.btn_node_config.setMinimumHeight(35)
        self.btn_node_config.setEnabled(False)
        self.btn_node_config.clicked.connect(self.open_node_placement_configurator)
        sidebar_layout.addWidget(self.btn_node_config)

        # Paper Add-on and PDF Export buttons (Requirement 3)
        self.btn_paper_addon = QPushButton("📄 PAPER ADD-ON")
        self.btn_paper_addon.setMinimumHeight(35)
        self.btn_paper_addon.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.btn_paper_addon.setEnabled(False)
        self.btn_paper_addon.clicked.connect(self.open_paper_addon)
        sidebar_layout.addWidget(self.btn_paper_addon)

        self.btn_pdf = QPushButton("📜 EXPORT BLUEPRINT REPORT (PDF)")
        self.btn_pdf.setMinimumHeight(35)
        self.btn_pdf.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        self.btn_pdf.setEnabled(False)
        self.btn_pdf.clicked.connect(self.export_integrated_pdf_report)
        sidebar_layout.addWidget(self.btn_pdf)

        legend_frame = QFrame()
        legend_frame.setObjectName("WidgetBox")
        legend_layout = QVBoxLayout(legend_frame)
        legend_title = QLabel("COLOR CODES LEGEND")
        legend_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        legend_title.setStyleSheet("background-color: #222E27; font-weight: bold;")
        legend_layout.addWidget(legend_title)
        
        items = [
            ("Original Green", "Passive Protected Areas"), 
            ("White", "Walkways"), 
            ("Pink", "Seating Units"), 
            ("Blue", "Playgrounds"), 
            ("Turquoise", "Water Features"), 
            ("Orange", "Sports Complexes"), 
            ("Black", "Entrance Gates")
        ]
        for icon, desc in items:
            lbl = QLabel(f"{icon}: {desc}")
            lbl.setFont(QFont("Segoe UI", 8))
            lbl.setStyleSheet("background-color: #222E27")
            legend_layout.addWidget(lbl)
        sidebar_layout.addWidget(legend_frame)
        
        splitter.addWidget(sidebar)

        # ---------------- CENTRAL VIEWPORT MATPLOTLIB CANVAS ----------------
        center_frame = QFrame()
        center_frame.setObjectName("DashboardFrame")
        center_layout = QVBoxLayout(center_frame)

        self.figure = plt.figure(figsize=(11, 9), dpi=140)
        self.figure.patch.set_facecolor("#EBEAE5")
        gs = self.figure.add_gridspec(2, 4, height_ratios=[1, 2.3])
        
        self.agent_axes = [self.figure.add_subplot(gs[0, j]) for j in range(4)]
        self.ax_total = self.figure.add_subplot(gs[1, :]) 
        
        for ax in self.agent_axes + [self.ax_total]:
            ax.set_facecolor("#FFFFFF")
            ax.set_xticks([])
            ax.set_yticks([])
            
        self.canvas = FigureCanvas(self.figure)
        center_layout.addWidget(self.canvas)
        splitter.addWidget(center_frame)

        # ---------------- RIGHT PANEL: RESULTS & ANALYTICS ----------------
        right_panel = QFrame()
        right_panel.setObjectName("DashboardFrame")
        right_panel_layout = QVBoxLayout(right_panel)
        right_panel_layout.setContentsMargins(10, 12, 10, 12)

        right_panel_layout.addWidget(QLabel("<h3>RESULTS & ANALYTICS</h3>"))
        
        self.result_table = QTableWidget(4, 9) 
        self.result_table.setHorizontalHeaderLabels(["Cohort", "Seat", "Play", "Water", "Sport", "Walk", "α", "β", "Count"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.result_table.setFixedHeight(175)
        right_panel_layout.addWidget(self.result_table)

        right_panel_layout.addSpacing(5)
        right_panel_layout.addWidget(QLabel("<b>📊 SPATIAL STATISTICS FIELD SUMMARY</b>"))

        scroll = QScrollArea()
        scroll.setObjectName("ResultsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background-color: transparent;")
        self.scroll_layout = QVBoxLayout(scroll_widget)
        self.scroll_layout.setSpacing(4)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        
        self.report_labels = {}
        agents_list = list(BASE_PREFS.keys()) + ["Collective"]

        for ag in agents_list:
            frame = QFrame()
            if ag == "Collective":
                frame.setStyleSheet("background-color: #E6EBE6; border: 1.5px solid #4A5D4E; border-radius: 6px;")
            frm_layout = QVBoxLayout(frame)
            frm_layout.setContentsMargins(5, 5, 5, 5)
            
            t_msg = "NASH WEIGHTED CONSENSUS CONVERGENCE" if ag == "Collective" else f"{ag.upper()} SATISFACTION INDEX"
            title = QLabel(t_msg)
            title.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            title.setStyleSheet("color: #4A5D4E;" if ag == "Collective" else f"color: {AGENT_COLORS[ag]};")
            frm_layout.addWidget(title)
            
            self.report_labels[f"{ag}_VH"] = QLabel("Very High (Red): %0.00")
            self.report_labels[f"{ag}_H"]  = QLabel("High (Orange): %0.00")
            self.report_labels[f"{ag}_M"]  = QLabel("Medium (Yellow): %0.00")
            self.report_labels[f"{ag}_L"]  = QLabel("Low (Lt Green): %0.00")
            self.report_labels[f"{ag}_VL"] = QLabel("Very Low (Dk Green): %0.00")
            
            for key in [f"{ag}_VH", f"{ag}_H", f"{ag}_M", f"{ag}_L", f"{ag}_VL"]:
                self.report_labels[key].setFont(QFont("Consolas", 8))
                frm_layout.addWidget(self.report_labels[key])
            self.scroll_layout.addWidget(frame)

        scroll.setWidget(scroll_widget)
        right_panel_layout.addWidget(scroll, stretch=2)

        self.graph_figure, self.graph_ax = plt.subplots(figsize=(3, 1.8))
        self.graph_figure.patch.set_facecolor("#ebeae5")
        self.graph_ax.set_facecolor("#FFFFFF")
        self.graph_ax.axis('off')
        self.graph_canvas = FigureCanvas(self.graph_figure)
        right_panel_layout.addWidget(self.graph_canvas, stretch=1)

        self.lbl_status = QLabel("Simulation Workspace Ready.")
        self.lbl_status.setStyleSheet("color: #4A5D4E; font-weight: bold; font-size: 11px;")
        right_panel_layout.addWidget(self.lbl_status)

        # Renamed run simulation button
        self.btn_run_sim = QPushButton("⚡ RUN SIMULATION ENGINE")
        self.btn_run_sim.setStyleSheet("background-color: #4A5D4E; color: white; height: 38px; border-radius: 6px; font-weight: bold;")
        self.btn_run_sim.setEnabled(False)
        self.btn_run_sim.clicked.connect(self.fire_async_simulation_worker)
        right_panel_layout.addWidget(self.btn_run_sim)

        splitter.addWidget(right_panel)
        splitter.setSizes([240, 960, 620])
        
        self.create_menu_bar()
        self.sync_and_recalculate_counts()
        self.render_canvas_plots()

    def create_menu_bar(self):
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background-color: #EBEAE5; color: #2D2D2D; border-bottom: 1px solid #D1CCC0; }
            QMenuBar::item { background-color: transparent; padding: 4px 10px; }
            QMenuBar::item:selected { background-color: #B8A88A; color: white; }
            QMenu { background-color: #EBEAE5; color: #2D2D2D; border: 1px solid #D1CCC0; }
            QMenu::item { padding: 4px 20px; }
            QMenu::item:selected { background-color: #B8A88A; color: white; }
        """)
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        self.act_load_layout = file_menu.addAction("Load Layout")
        self.act_load_layout.triggered.connect(self.load_image_and_analyze)
        
        self.act_import_data = file_menu.addAction("Import Data")
        self.act_import_data.triggered.connect(self.import_complete_project_from_excel)
        self.act_import_data.setEnabled(False)
        
        self.act_export_data = file_menu.addAction("Export Data")
        self.act_export_data.triggered.connect(self.export_complete_project_to_excel)
        self.act_export_data.setEnabled(False)
        
        file_menu.addSeparator()
        self.act_quit = file_menu.addAction("Quit")
        self.act_quit.triggered.connect(self.close)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        self.act_pref_matrix = edit_menu.addAction("Preference Matrix")
        self.act_pref_matrix.triggered.connect(self.open_parameter_menu)
        self.act_pref_matrix.setEnabled(False)
        
        self.act_node_config = edit_menu.addAction("Node Placement Configurator")
        self.act_node_config.triggered.connect(self.open_node_placement_configurator)
        self.act_node_config.setEnabled(False)
        
        # Simulation Engine menu
        sim_menu = menubar.addMenu("Simulation Engine")
        
        self.act_run_sim = sim_menu.addAction("Run Simulation Engine")
        self.act_run_sim.triggered.connect(self.fire_async_simulation_worker)
        self.act_run_sim.setEnabled(False)
        
        # Reports menu
        reports_menu = menubar.addMenu("Reports")
        
        self.act_paper_addon = reports_menu.addAction("PAPER ADD-ON")
        self.act_paper_addon.triggered.connect(self.open_paper_addon)
        self.act_paper_addon.setEnabled(False)
        
        self.act_pdf_blueprint = reports_menu.addAction("Pdf Blueprint")
        self.act_pdf_blueprint.triggered.connect(self.export_integrated_pdf_report)
        self.act_pdf_blueprint.setEnabled(False)
        
        # About menu
        self.act_about = menubar.addAction("About")
        self.act_about.triggered.connect(self.open_about_dialog)

    def open_about_dialog(self):
        QMessageBox.about(
            self,
            "About Nash Equilibrium Landscape Simulator (NELS)",
            "<p>Nash Equilibrium Landscape Simulator (NELS) is an advanced spatial decision support system "
            "designed for landscape architecture, urban design, and environmental planning. By bridging the gap "
            "between architectural vision and computational intelligence, NELS utilizes a multi-criterion \"Nash "
            "Equilibrium Model\" to simulate the complex spatial preferences of diverse user cohorts—including youth, "
            "the elderly, parents, and athletes—within urban and recreational environments.</p>"
            "<p>Moving beyond traditional design methodologies, NELS empowers professionals to conduct sophisticated "
            "spatial density analyses and weighted consensus calculations. It effectively transforms qualitative design "
            "goals into quantitative, data-driven strategies, ensuring the highest level of user satisfaction through "
            "optimized spatial configurations.</p>"
            "<p>Built on a robust Python-based engine and integrated with professional analytical tools, NELS streamlines "
            "the design workflow by allowing seamless synchronization with external data. This provides designers not "
            "only with visual design tools but also with a comprehensive suite of statistical insights—including "
            "heatmaps, analytical reports, and high-resolution blueprint documentation—grounding every design decision "
            "in scientific rigor.</p>"
            "<p>PLEASE USE THE COLOR INFORMATON TO CREATE MAP =<p>" 
            "<p>WALKWAY: [255, 255, 255],<br>"     
            "SEATING: [255, 192, 203],<br>"     
            "PLAYGROUND: [0, 0, 255],<br>"      
            "WATER: [0, 255, 255],<br>"          
            "SPORTS: [255, 165, 0],<br>"         
            "GREEN: [0, 128, 0],<br>"            
            "ENTRANCE: [0, 0, 0]</p>"
            "<p>contact: benliay@akdeniz.edu.tr</p>"
        )

    def open_node_placement_configurator(self):
        if not self.map_loaded:
            QMessageBox.warning(self, "Warning", "Upload layout backdrop template image first!")
            return
        dialog = NodePlacementConfiguratorDialog(self, self)
        dialog.exec()
        self.render_canvas_plots()
        if dialog.transfer_pressed:
            self.fire_async_simulation_worker()

    def sync_and_recalculate_counts(self):
        self.update_live_matrix_table()

    def invalidate_simulation_matrices(self):
        self.nash_analyzed = False
        self.latest_nash_matrix = None
        self.latest_density_rgb = None
        self.btn_paper_addon.setEnabled(False)
        self.act_paper_addon.setEnabled(False)
        self.lbl_status.setText("Workspace modified. Sync values and run sim framework.")

    def load_image_and_analyze(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Design Layout Template", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            val, ok = QInputDialog.getDouble(self, "Scale Factor Parameters", 
                                             "How many pixels correspond to 1 Meter?\n(e.g., if 1 meter equals 2 pixels, enter 2.0)", 
                                             value=1.0, min=0.001, max=1000.0, decimals=4)
            if not ok: val = 1.0
            
            try:
                with open(file_name, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"Could not read/decode the image file at: {file_name}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                dialog = AreaSelectionDialog(img, self)
                if dialog.exec() == QDialog.DialogCode.Accepted and dialog.crop_coords is not None:
                    x1, y1, x2, y2 = dialog.crop_coords
                    if (x2 - x1) > 2 and (y2 - y1) > 2:
                        img = img[y1:y2, x1:x2]
                
                w, h = self.evaluator.load_design_image(img, val)
            except Exception as e:
                QMessageBox.critical(self, "Image Load Error", f"Failed to load design template:\n{str(e)}")
                return
                
            self.map_loaded = True
            self.placed_agents.clear()
            self.invalidate_simulation_matrices()
            
            self.lbl_dimensions.setText(f"<b>Workspace Bounds:</b> {w} x {h} Pixels (Scale: {val:.2f} px/m)")
            self.lbl_dimensions.setStyleSheet("background-color: transparent; color: #27AE60; font-size: 11px;")
            
            w_meters = w / val
            h_meters = h / val
            
            self.input_ax.clear()
            self.input_ax.imshow(self.evaluator.grid, extent=[0, w_meters, h_meters, 0])
            self.input_ax.axis('off')
            self.input_canvas.draw()
            
            self.btn_menu.setEnabled(True)
            self.btn_node_config.setEnabled(True)
            self.btn_run_sim.setEnabled(True)
            self.btn_pdf.setEnabled(True)
            
            # Enable Menu Actions
            self.act_import_data.setEnabled(True)
            self.act_export_data.setEnabled(True)
            self.act_pref_matrix.setEnabled(True)
            self.act_node_config.setEnabled(True)
            self.act_run_sim.setEnabled(True)
            self.act_pdf_blueprint.setEnabled(True)
            
            self.sync_and_recalculate_counts()
            self.render_canvas_plots()
            self.fire_async_simulation_worker()

    def open_parameter_menu(self):
        dialog = ParameterDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.update_live_matrix_table()
            self.invalidate_simulation_matrices()
            self.fire_async_simulation_worker()

    def update_live_matrix_table(self):
        total_agents = len(self.placed_agents)
        self.result_table.setRowCount(0)
        
        for idx, ag in enumerate(AGENTS_LIST):
            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(ag))
            
            live_count = sum(1 for a in self.placed_agents if a['type'] == ag)
            
            if total_agents > 0:
                p_ratio = live_count / total_agents
                for u_idx, unit in enumerate(UNITS_LIST):
                    self.result_table.setItem(row, u_idx + 1, QTableWidgetItem(f"{(BASE_PREFS[ag][unit] * p_ratio):.1f}"))
                self.result_table.setItem(row, 6, QTableWidgetItem(f"{(BASE_PREFS[ag]['alpha'] * p_ratio):.2f}"))
                self.result_table.setItem(row, 7, QTableWidgetItem(f"{(BASE_PREFS[ag]['beta'] * p_ratio):.2f}"))
                self.result_table.setItem(row, 8, QTableWidgetItem(f"{live_count}"))
            else:
                for u_idx, unit in enumerate(UNITS_LIST):
                    self.result_table.setItem(row, u_idx + 1, QTableWidgetItem(f"{(AGENT_PREFS[ag][unit]):.1f}"))
                self.result_table.setItem(row, 6, QTableWidgetItem(f"{AGENT_PREFS[ag]['alpha']:.2f}"))
                self.result_table.setItem(row, 7, QTableWidgetItem(f"{AGENT_PREFS[ag]['beta']:.2f}"))
                self.result_table.setItem(row, 8, QTableWidgetItem(f"{int(AGENT_PREFS[ag]['count'])}"))

    # =====================================================================
    # SPREADSHEET STORAGE PLATFORM
    # =====================================================================
    def export_complete_project_to_excel(self):
        if not self.placed_agents:
            QMessageBox.warning(self, "Warning", "Place simulation coordinate nodes before executing parameters logging.")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Export Complete Project Package", "Simulation_Project_Data.xlsx", "Excel Files (*.xlsx)")
        if not save_path: return
        try:
            df_nodes = pd.DataFrame(self.placed_agents)
            df_nodes = df_nodes[['type', 'x', 'y']].rename(columns={'type': 'Agent_Type', 'x': 'Coordinate_X', 'y': 'Coordinate_Y'})
            
            matrix_records = []
            for ag in AGENTS_LIST:
                rec = {"Cohort_Profile": ag}
                rec.update({unit: AGENT_PREFS[ag][unit] for unit in UNITS_LIST})
                rec["Alpha_Penalty"] = AGENT_PREFS[ag]["alpha"]
                rec["Beta_Penalty"] = AGENT_PREFS[ag]["beta"]
                rec["Active_Count"] = AGENT_PREFS[ag]["count"] 
                matrix_records.append(rec)
            df_matrix = pd.DataFrame(matrix_records)

            with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
                df_nodes.to_excel(writer, sheet_name='Spatial_Node_Coordinates', index=False)
                df_matrix.to_excel(writer, sheet_name='Preference_Weights_Matrix', index=False)
                
            QMessageBox.information(self, "Success", "Project archive packages built successfully!")
        except Exception as e:
            QMessageBox.critical(self, "IO Error", f"Failed to successfully serialize archive packages data sheets:\n{str(e)}")

    def import_complete_project_from_excel(self):
        if not self.map_loaded:
            QMessageBox.warning(self, "Warning", "Load layout backdrop blueprint map prior to calling excel streams.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Complete Project Package Spreadsheet", "", "Excel Files (*.xlsx)")
        if not file_path: return
        try:
            excel_file = pd.ExcelFile(file_path)
            if 'Spatial_Node_Coordinates' not in excel_file.sheet_names or 'Preference_Weights_Matrix' not in excel_file.sheet_names:
                QMessageBox.critical(self, "Validation Error", "Target archive file missing structural sheets indices.")
                return
                
            df_nodes = pd.read_excel(file_path, sheet_name='Spatial_Node_Coordinates')
            df_matrix = pd.read_excel(file_path, sheet_name='Preference_Weights_Matrix')
            
            for _, row in df_matrix.iterrows():
                ag = str(row['Cohort_Profile'])
                if ag in AGENT_PREFS:
                    for unit in UNITS_LIST:
                        AGENT_PREFS[ag][unit] = float(row[unit])
                    AGENT_PREFS[ag]["alpha"] = float(row["Alpha_Penalty"])
                    AGENT_PREFS[ag]["beta"] = float(row["Beta_Penalty"])
                    AGENT_PREFS[ag]["count"] = float(row["Active_Count"]) 
            
            self.placed_agents.clear()
            self.invalidate_simulation_matrices()
            
            for _, row in df_nodes.iterrows():
                a_type = str(row['Agent_Type'])
                x_val = int(row['Coordinate_X'])
                y_val = int(row['Coordinate_Y'])
                if 0 <= x_val < self.evaluator.w and 0 <= y_val < self.evaluator.h:
                    if a_type in AGENTS_LIST:
                        self.placed_agents.append({'type': a_type, 'x': x_val, 'y': y_val})
                        
            self.sync_and_recalculate_counts()
            self.render_canvas_plots()
            self.fire_async_simulation_worker()
            QMessageBox.information(self, "Success", f"Project repository parsed. Restored {len(self.placed_agents)} structural nodes.")
        except Exception as e:
            QMessageBox.critical(self, "Parsing Error", f"Failed to extract parameter maps from destination sheets profiles:\n{str(e)}")

    # =====================================================================
    # MULTITHREADED ANALYTICAL ENGINES CONTROLS
    # =====================================================================
    def fire_async_simulation_worker(self):
        self.btn_run_sim.setText("🔄 INTERSECTING...")
        self.btn_run_sim.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.act_run_sim.setEnabled(False)
        
        for ax in self.agent_axes:
            ax.clear()
            ax.axis('off')
        self.ax_total.clear()
        self.ax_total.axis('off')
        self.ax_total.text(0.5, 0.5, "Simulation is running...", color='#4A5D4E', 
                           ha='center', va='center', fontsize=14, fontweight='bold')
        self.canvas.draw()
        
        self.worker = AnalysisWorker(self.evaluator, self.placed_agents)
        self.worker.progress_changed.connect(self.on_simulation_progress_update)
        self.worker.finished_with_result.connect(self.on_simulation_converged)
        self.worker.start()

    def on_simulation_progress_update(self, percentage, text_msg):
        self.lbl_status.setText(f"<b>[{percentage}%]</b> {text_msg}")

    def on_simulation_converged(self, initial_utility, nash_maps, collective, gate_dist, density_rgb):
        self.latest_nash_matrix = collective
        self.latest_density_rgb = density_rgb
        self.latest_nash_maps = nash_maps
        self.nash_analyzed = True
        
        self.current_report_data = {}
        
        # 1. Setup overlay colors: Yeşil alanlar -> Koyu gri, Analiz dışı alanlar -> Açık gri
        overlay = np.zeros((self.evaluator.h, self.evaluator.w, 4))
        # Green spaces -> Dark Gray (#555555 / 0.33)
        overlay[self.evaluator.green_mask, :3] = 0.33
        overlay[self.evaluator.green_mask, 3] = 1.0
        # Out of bounds (Masked gray) -> Light Gray (#D3D3D3 / 0.83)
        overlay[self.evaluator.masked_gray_mask, :3] = 0.83
        overlay[self.evaluator.masked_gray_mask, 3] = 1.0

        w_meters = self.evaluator.w / self.evaluator.px_per_meter
        h_meters = self.evaluator.h / self.evaluator.px_per_meter

        for i, agent in enumerate(AGENTS_LIST):
            ax = self.agent_axes[i]
            ax.clear()
            data = nash_maps[agent]
            data_masked = np.ma.masked_where((data == -np.inf) | self.evaluator.green_mask, data)
            
            ax.imshow(data_masked, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
            ax.imshow(overlay, extent=[0, w_meters, h_meters, 0])
            self.format_subplot_axes_grid(ax, f"LAYER: {agent.upper()}", 7)
            
            vh, h, m, l, vl = self.calculate_scale_percentages(data)
            self.report_labels[f"{agent}_VH"].setText(f"Very High (Red): %{vh:.2f}")
            self.report_labels[f"{agent}_H"].setText(f"High (Orange): %{h:.2f}")
            self.report_labels[f"{agent}_M"].setText(f"Medium (Yellow): %{m:.2f}")
            self.report_labels[f"{agent}_L"].setText(f"Low (Lt Green): %{l:.2f}")
            self.report_labels[f"{agent}_VL"].setText(f"Very Low (Dk Green): %{vl:.2f}")
            self.current_report_data[agent] = {"VH": vh, "H": h, "M": m, "L": l, "VL": vl, "matrix": data_masked}

        self.ax_total.clear()
        collective_masked = np.ma.masked_where((collective == 0) | self.evaluator.green_mask, collective)
        self.ax_total.imshow(collective_masked, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
        self.ax_total.imshow(overlay, extent=[0, w_meters, h_meters, 0])
        self.format_subplot_axes_grid(self.ax_total, "OPTIMIZATION: WEIGHTED EQUILIBRIUM", 9)

        vh, h, m, l, vl = self.calculate_scale_percentages(collective)
        self.report_labels["Collective_VH"].setText(f"Very High (Red): %{vh:.2f}")
        self.report_labels["Collective_H"].setText(f"High (Orange): %{h:.2f}")
        self.report_labels["Collective_M"].setText(f"Medium (Yellow): %{m:.2f}")
        self.report_labels["Collective_L"].setText(f"Low (Lt Green): %{l:.2f}")
        self.report_labels["Collective_VL"].setText(f"Very Low (Dk Green): %{vl:.2f}")
        self.current_report_data["Collective"] = {"VH": vh, "H": h, "M": m, "L": l, "VL": vl, "matrix": collective_masked}

        self.graph_ax.clear()
        self.graph_ax.axis('on')
        self.graph_ax.set_facecolor("#FFFFFF")
        intervals = ['V.High', 'High', 'Medium', 'Low', 'V.Low']
        self.graph_ax.bar(intervals, [vh, h, m, l, vl], color=BAR_RENKLERI_CONSENSUS, width=0.55)
        self.graph_ax.set_title("NASH WEIGHTED CONSENSUS AREA (%)", color="#4A5D4E", fontsize=7, fontweight='bold')
        self.graph_ax.tick_params(colors='#2D2D2D', labelsize=6)
        for spine in self.graph_ax.spines.values(): spine.set_color('#D1CCC0')
            
        self.graph_figure.tight_layout()
        self.graph_canvas.draw()
        
        self.btn_run_sim.setText("⚡ RUN SIMULATION ENGINE")
        self.btn_run_sim.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.btn_paper_addon.setEnabled(True)
        self.lbl_status.setText("✅ Multi-distance simulation equilibrium solved successfully.")
        
        self.act_run_sim.setEnabled(True)
        self.act_paper_addon.setEnabled(True)
        
        self.render_canvas_plots()

    def map_loaded_status(self):
        return self.map_loaded

    def format_subplot_axes_grid(self, ax, heading_str, size_font):
        ax.set_facecolor("#FFFFFF")
        ax.set_title(heading_str, fontsize=size_font, fontweight='bold', color="#4A5D4E")
        
        # Calculate dynamic spacing in meters based on the current scale
        w_meters = self.evaluator.w / self.evaluator.px_per_meter
        spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
        ax.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.5, alpha=1.0)
        ax.tick_params(colors='#2D2D2D', labelsize=6)
        for spine in ax.spines.values(): spine.set_color('#D1CCC0')

    def calculate_scale_percentages(self, data_matrix):
        valid_pixels = data_matrix[(data_matrix != -np.inf) & (~self.evaluator.green_mask)]
        valid_pixels = valid_pixels[valid_pixels > 0]
        if len(valid_pixels) > 0:
            min_v, max_v = valid_pixels.min(), valid_pixels.max()
            skala = max_v - min_v if (max_v - min_v) > 0 else 1
            e1 = min_v + (skala / 5)
            e2 = min_v + (2 * skala / 5)
            e3 = min_v + (3 * skala / 5)
            e4 = min_v + (4 * skala / 5)
            
            v_low_c = np.sum(valid_pixels <= e1)
            low_c   = np.sum((valid_pixels > e1) & (valid_pixels <= e2))
            mid_c   = np.sum((valid_pixels > e2) & (valid_pixels <= e3))
            high_c  = np.sum((valid_pixels > e3) & (valid_pixels <= e4))
            v_high_c= np.sum(valid_pixels > e4)
            total = len(valid_pixels)
            return (v_high_c/total)*100, (high_c/total)*100, (mid_c/total)*100, (low_c/total)*100, (v_low_c/total)*100
        return 0.0, 0.0, 0.0, 0.0, 0.0

    def render_canvas_plots(self):
        self.ax_total.clear()
        self.ax_total.set_facecolor('#FFFFFF')
        
        w_meters = self.evaluator.w / self.evaluator.px_per_meter
        h_meters = self.evaluator.h / self.evaluator.px_per_meter
        
        if self.map_loaded:
            if not self.nash_analyzed:
                self.ax_total.imshow(self.evaluator.display_grid, extent=[0, w_meters, h_meters, 0])
                self.format_subplot_axes_grid(self.ax_total, "OPTIMIZATION: WEIGHTED EQUILIBRIUM (Run simulation to solve)", 9)
            else:
                if self.latest_nash_matrix is not None:
                    overlay = np.zeros((self.evaluator.h, self.evaluator.w, 4))
                    overlay[self.evaluator.green_mask, :3] = 0.33
                    overlay[self.evaluator.green_mask, 3] = 1.0
                    overlay[self.evaluator.masked_gray_mask, :3] = 0.83
                    overlay[self.evaluator.masked_gray_mask, 3] = 1.0
                    
                    collective_masked = np.ma.masked_where((self.latest_nash_matrix == 0) | self.evaluator.green_mask, self.latest_nash_matrix)
                    self.ax_total.imshow(collective_masked, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
                    self.ax_total.imshow(overlay, extent=[0, w_meters, h_meters, 0])
                    self.format_subplot_axes_grid(self.ax_total, "OPTIMIZATION: WEIGHTED EQUILIBRIUM", 9)
                
                for i, agent in enumerate(AGENTS_LIST):
                    ax = self.agent_axes[i]
                    ax.clear()
                    if agent in self.latest_nash_maps:
                        data = self.latest_nash_maps[agent]
                        data_masked = np.ma.masked_where((data == -np.inf) | self.evaluator.green_mask, data)
                        
                        overlay = np.zeros((self.evaluator.h, self.evaluator.w, 4))
                        overlay[self.evaluator.green_mask, :3] = 0.33
                        overlay[self.evaluator.green_mask, 3] = 1.0
                        overlay[self.evaluator.masked_gray_mask, :3] = 0.83
                        overlay[self.evaluator.masked_gray_mask, 3] = 1.0
                        
                        ax.imshow(data_masked, cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
                        ax.imshow(overlay, extent=[0, w_meters, h_meters, 0])
                        self.format_subplot_axes_grid(ax, f"LAYER: {agent.upper()}", 7)
        else:
            self.ax_total.text(0.5, 0.5, "Upload layout backdrop template image...", color='gray', ha='center', va='center')
            self.ax_total.axis('off')
            
            for ax in self.agent_axes:
                ax.clear()
                ax.axis('off')
                
        self.figure.tight_layout()
        self.canvas.draw()

    def run_simulation_synchronously(self):
        density_rgb = self.evaluator.compute_spatial_density_field(self.placed_agents)
        gate_dist_map = self.evaluator.calculate_gate_distance_map()
        all_use_dist_maps = self.evaluator.calculate_all_use_distance_maps()
        
        agent_utility_maps = {}
        total_population = sum(params["count"] for params in AGENT_PREFS.values())
        if total_population == 0: total_population = 1
        
        lambda_decay = 0.08 

        for agent, params in AGENT_PREFS.items():
            u_map = np.zeros((self.evaluator.h, self.evaluator.w))
            for y in range(self.evaluator.h):
                for x in range(self.evaluator.w):
                    area = self.evaluator.get_area_type(self.evaluator.grid[y, x])
                    if area == "MASKED_GRAY" or self.evaluator.green_mask[y, x]:
                        u_map[y, x] = -np.inf
                        continue
                    net_utility = 0.0
                    for other_area, weight in params.items():
                        if other_area in ["alpha", "beta", "count", "GREEN"]:
                            continue
                        dist_to_use = all_use_dist_maps[other_area][y, x]
                        if weight == -999.0:
                            if dist_to_use == 0: 
                                net_utility = -np.inf
                                break
                        else:
                            net_utility += weight * np.exp(-lambda_decay * dist_to_use)
                    if net_utility != -np.inf:
                        u_map[y, x] = net_utility - (params["alpha"] * gate_dist_map[y, x])
                    else:
                        u_map[y, x] = -np.inf
            agent_utility_maps[agent] = u_map

        nash_satisfaction = {agent: u_map.copy() for agent, u_map in agent_utility_maps.items()}
        
        total_loops = 5
        for loop in range(total_loops):
            total_occupancy = np.zeros((self.evaluator.h, self.evaluator.w))
            for agent, params in AGENT_PREFS.items():
                normalized = np.maximum(0, nash_satisfaction[agent])
                if normalized.max() > 0:
                    cohort_weight = (params["count"] / total_population) * 4.0
                    total_occupancy += (normalized / normalized.max()) * cohort_weight

            for agent, params in AGENT_PREFS.items():
                nash_satisfaction[agent] = agent_utility_maps[agent] - (params["beta"] * total_occupancy)
                nash_satisfaction[agent][agent_utility_maps[agent] == -np.inf] = -np.inf

        collective_map = np.zeros((self.evaluator.h, self.evaluator.w))
        total_weight_grid = np.zeros((self.evaluator.h, self.evaluator.w))
        
        for agent, params in AGENT_PREFS.items():
            valid_mask = nash_satisfaction[agent] != -np.inf
            weight = params["count"]
            collective_map[valid_mask] += nash_satisfaction[agent][valid_mask] * weight
            total_weight_grid[valid_mask] += weight
            
        collective_map = np.divide(collective_map, total_weight_grid, out=np.zeros_like(collective_map), where=total_weight_grid!=0)
        
        self.latest_nash_matrix = collective_map
        self.latest_density_rgb = density_rgb
        self.latest_nash_maps = nash_satisfaction
        self.nash_analyzed = True
        
        self.current_report_data = {}
        for agent in AGENTS_LIST:
            data = nash_satisfaction[agent]
            data_masked = np.ma.masked_where((data == -np.inf) | self.evaluator.green_mask, data)
            vh, h, m, l, vl = self.calculate_scale_percentages(data)
            self.current_report_data[agent] = {"VH": vh, "H": h, "M": m, "L": l, "VL": vl, "matrix": data_masked}
            
        collective_masked = np.ma.masked_where((collective_map == 0) | self.evaluator.green_mask, collective_map)
        vh, h, m, l, vl = self.calculate_scale_percentages(collective_map)
        self.current_report_data["Collective"] = {"VH": vh, "H": h, "M": m, "L": l, "VL": vl, "matrix": collective_masked}

    # =====================================================================
    # 6. REPORTLAB BLUEPRINT EXPORT COMPILED ENGINE
    # =====================================================================
    def export_integrated_pdf_report(self):
        if not self.map_loaded: return
        
        if not self.nash_analyzed:
            if self.placed_agents:
                total_agents = len(self.placed_agents)
                for idx, ag in enumerate(AGENTS_LIST):
                    live_count = sum(1 for a in self.placed_agents if a['type'] == ag)
                    p_ratio = live_count / total_agents if total_agents > 0 else 0.0
                    for unit in UNITS_LIST:
                        AGENT_PREFS[ag][unit] = float(BASE_PREFS[ag][unit] * p_ratio)
                    AGENT_PREFS[ag]["alpha"] = float(BASE_PREFS[ag]["alpha"] * p_ratio)
                    AGENT_PREFS[ag]["beta"] = float(BASE_PREFS[ag]["beta"] * p_ratio)
                    AGENT_PREFS[ag]["count"] = float(live_count)
                self.update_live_matrix_table()
            self.run_simulation_synchronously()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Layout Report", "Integrated_Landscape_Full_Report.pdf", "PDF Files (*.pdf)")
        if not file_path: return

        tmp_dir = os.path.dirname(file_path)
        img_paths = {}
        
        w_meters = self.evaluator.w / self.evaluator.px_per_meter
        h_meters = self.evaluator.h / self.evaluator.px_per_meter

        if self.placed_agents or self.nash_analyzed:
            main_img_p = os.path.join(tmp_dir, "tmp_pdf_main_canvas.png")
            fig_m, ax_m = plt.subplots(figsize=(6, 5))
            fig_m.patch.set_facecolor("#ebeae5")
            ax_m.imshow(self.evaluator.display_grid, extent=[0, w_meters, h_meters, 0])
            for agent in self.placed_agents:
                ax_m.scatter(agent['x'] / self.evaluator.px_per_meter, 
                             agent['y'] / self.evaluator.px_per_meter, 
                             color=AGENT_COLORS[agent['type']], edgecolors='white', s=45, linewidths=0.7, zorder=10)
            ax_m.set_xlim(0, w_meters)
            ax_m.set_ylim(h_meters, 0)
            ax_m.axis('off')
            fig_m.tight_layout()
            fig_m.savefig(main_img_p, facecolor=fig_m.get_facecolor(), dpi=180, bbox_inches='tight')
            plt.close(fig_m)
            img_paths["MainGrid"] = main_img_p

        if self.nash_analyzed:
            satisfaction_overlay = np.zeros((self.evaluator.h, self.evaluator.w, 4))
            satisfaction_overlay[self.evaluator.green_mask, :3] = 0.33
            satisfaction_overlay[self.evaluator.green_mask, 3] = 1.0
            satisfaction_overlay[self.evaluator.masked_gray_mask, :3] = 0.83
            satisfaction_overlay[self.evaluator.masked_gray_mask, 3] = 1.0

            for ag in AGENTS_LIST + ["Collective"]:
                fig, ax = plt.subplots(figsize=(6, 5))
                fig.patch.set_facecolor("#FFFFFF")
                ax.set_facecolor("#FFFFFF")
                ax.imshow(self.current_report_data[ag]["matrix"], cmap=CMAP_CONTINUOUS_SATISFACTION, extent=[0, w_meters, h_meters, 0])
                ax.imshow(satisfaction_overlay, extent=[0, w_meters, h_meters, 0])
                ax.axis('off')
                fig.tight_layout()
                p = os.path.join(tmp_dir, f"tmp_pdf_{ag}.png")
                fig.savefig(p, facecolor=fig.get_facecolor(), dpi=140, bbox_inches='tight')
                plt.close(fig)
                img_paths[ag] = p

            if self.placed_agents:
                dens_p = os.path.join(tmp_dir, "tmp_pdf_density.png")
                fig_d, ax_d = plt.subplots(figsize=(6, 5))
                fig_d.patch.set_facecolor("#FFFFFF")
                ax_d.set_facecolor("#FFFFFF")
                ax_d.imshow(self.latest_density_rgb, extent=[0, w_meters, h_meters, 0])
                ax_d.imshow(satisfaction_overlay, extent=[0, w_meters, h_meters, 0])
                spacing = max(5, int(round(w_meters / 10.0 / 5.0)) * 5)
                ax_d.xaxis.set_major_locator(ticker.MultipleLocator(spacing))
                ax_d.yaxis.set_major_locator(ticker.MultipleLocator(spacing))
                ax_d.grid(True, which='major', color='#D1CCC0', linestyle=':', linewidth=0.5, alpha=1.0)
                ax_d.tick_params(colors='#2D2D2D', labelsize=6)
                for spine in ax_d.spines.values(): spine.set_color('#D1CCC0')
                fig_d.tight_layout()
                fig_d.savefig(dens_p, facecolor=fig_d.get_facecolor(), dpi=150, bbox_inches='tight')
                plt.close(fig_d)
                img_paths["Density"] = dens_p

        doc = SimpleDocTemplate(file_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []
        styles = getSampleStyleSheet()
        
        FULL_PAGE_WIDTH = 532 
        
        t_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=20, spaceAfter=15, alignment=1, textColor=colors.HexColor("#2C3E50"))
        h2_style = ParagraphStyle('SecHeader', parent=styles['Heading2'], fontSize=12, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#16A085"))
        p_style = ParagraphStyle('ParaText', parent=styles['Normal'], fontSize=9.5, spaceAfter=10, leading=13)
        black_style = ParagraphStyle('BText', parent=styles['Normal'], fontSize=8.5, textColor=colors.black)
        bold_style = ParagraphStyle('BldText', parent=styles['Normal'], fontSize=8.5, fontName="Helvetica-Bold", textColor=colors.black)

        story.append(Paragraph("Spatial Optimization & Micro-Agent Report", t_style))
        story.append(Paragraph(f"<b>Report Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')} | <b>Grid Bounds:</b> {self.evaluator.w}x{self.evaluator.h} Px", p_style))
        story.append(Spacer(1, 10))
        
        story.append(Paragraph("1. Micro-Population Point Allocation Metrics", h2_style))
        total_p = len(self.placed_agents) if len(self.placed_agents) > 0 else 1
        
        t_pop_matrix = [[Paragraph('<b>User Profile Group Cohort</b>', bold_style), Paragraph('<b>Placed Nodes</b>', bold_style), Paragraph('<b>Proportional Area Ratio</b>', bold_style)]]
        for ag in AGENTS_LIST:
            t_pop_matrix.append([Paragraph(f"{ag} Cohort Model", black_style), f"{int(AGENT_PREFS[ag]['count'])} nodes", f"%{((AGENT_PREFS[ag]['count']/total_p)*100):.1f}"])
        t_pop_matrix.append([Paragraph("<b>Total Active Placed Sample</b>", bold_style), f"<b>{len(self.placed_agents)} nodes</b>", "<b>%100.0</b>"])
        
        t_pop = RLTable(t_pop_matrix, colWidths=[200, 166, 166])
        t_pop.setStyle(RLTableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F5F6FA")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DCDDE1")), ('PADDING', (0,0), (-1,-1), 4)]))
        story.append(t_pop)
        
        story.append(Spacer(1, 12))
        story.append(Paragraph("<b>Integrated Attraction / Repulsion Preference Matrix Data</b>", h2_style))
        
        t_pref_header = [Paragraph('<b>Cohort</b>', bold_style), Paragraph('<b>Seat</b>', bold_style), Paragraph('<b>Play</b>', bold_style), 
                         Paragraph('<b>Water</b>', bold_style), Paragraph('<b>Sport</b>', bold_style), Paragraph('<b>Walk</b>', bold_style), 
                         Paragraph('<b>Alpha (α)</b>', bold_style), Paragraph('<b>Beta (β)</b>', bold_style), Paragraph('<b>Count</b>', bold_style)]
        t_pref_matrix = [t_pref_header]
        
        for ag in AGENTS_LIST:
            t_pref_matrix.append([
                Paragraph(ag, black_style),
                f"{AGENT_PREFS[ag]['SEATING']:.1f}", f"{AGENT_PREFS[ag]['PLAYGROUND']:.1f}",
                f"{AGENT_PREFS[ag]['WATER']:.1f}", f"{AGENT_PREFS[ag]['SPORTS']:.1f}",
                f"{AGENT_PREFS[ag]['WALKWAY']:.1f}", f"{AGENT_PREFS[ag]['alpha']:.2f}",
                f"{AGENT_PREFS[ag]['beta']:.2f}", f"{int(AGENT_PREFS[ag]['count'])}"
            ])
            
        t_pref_tbl = RLTable(t_pref_matrix, colWidths=[82, 53, 53, 53, 53, 53, 64, 64, 59])
        t_pref_tbl.setStyle(RLTableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#EAEDED")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#BDC3C7")), ('PADDING', (0,0), (-1,-1), 5)]))
        story.append(t_pref_tbl)
        
        story.append(Spacer(1, 15))
        map_aspect = self.evaluator.h / self.evaluator.w
        if map_aspect <= 0.8:
            img_w = FULL_PAGE_WIDTH
            img_h = FULL_PAGE_WIDTH * map_aspect
        else:
            img_h = 420
            img_w = 420 / map_aspect
            
        if self.placed_agents or self.nash_analyzed:
            story.append(PageBreak())
            story.append(Paragraph("2. Workspace Core Layout Grid Blueprint", h2_style))
            story.append(RLImage(img_paths["MainGrid"], width=img_w, height=img_h))

        if self.nash_analyzed and self.placed_agents:
            story.append(PageBreak())
            story.append(Paragraph("3. Geostatistical Continuous Spatial Density Mapping", h2_style))
            story.append(Paragraph("Continuous surface tracking density hotspots combined with all active green and dark passive layout contour constraints.", p_style))
            story.append(RLImage(img_paths["Density"], width=img_w, height=img_h))
            
        if self.nash_analyzed:
            for ag in AGENTS_LIST:
                story.append(PageBreak())
                story.append(Paragraph(f"Analytical Layer: {ag.upper()} Proportional Model Framework", h2_style))
                story.append(RLImage(img_paths[ag], width=img_w, height=img_h))
                story.append(Spacer(1, 8))
                
                data = self.current_report_data[ag]
                t_ag_matrix = [
                    [Paragraph('<b>Satisfaction Scale Threshold</b>', bold_style), Paragraph('<b>Area Percentage Ratio</b>', bold_style)],
                    [Paragraph('Very High (Red Spectrum)', black_style), f"%{data['VH']:.2f}"],
                    [Paragraph('High (Orange Band)', black_style), f"%{data['H']:.2f}"],
                    [Paragraph('Medium (Yellow Segment)', black_style), f"%{data['M']:.2f}"],
                    [Paragraph('Low (Light Green Zone)', black_style), f"%{data['L']:.2f}"],
                    [Paragraph('Very Low (Dark Green Hub)', black_style), f"%{data['VL']:.2f}"]
                ]
                t_tbl = RLTable(t_ag_matrix, colWidths=[280, 252])
                t_tbl.setStyle(RLTableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#F5F6FA")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DCDDE1")), ('PADDING', (0,0), (-1,-1), 4)]))
                story.append(t_tbl)

            story.append(PageBreak())
            story.append(Paragraph("4. Nash Equilibrium Weighted Collective Consensus Mapping", h2_style))
            story.append(RLImage(img_paths["Collective"], width=img_w, height=img_h))
            story.append(Spacer(1, 8))
            
            c_data = self.current_report_data["Collective"]
            t_col_matrix = [
                [Paragraph('<b>Consensus Compromise Spectrum</b>', bold_style), Paragraph('<b>Area Slices Percentage</b>', bold_style)],
                [Paragraph('Very High (Red Zone - Peak Optimization Target)', black_style), f"%{c_data['VH']:.2f}"],
                [Paragraph('High (Orange Zone)', black_style), f"%{c_data['H']:.2f}"],
                [Paragraph('Medium (Yellow Solution Space)', black_style), f"%{c_data['M']:.2f}"],
                [Paragraph('Low (Light Green Zone)', black_style), f"%{c_data['L']:.2f}"],
                [Paragraph('Very Low (Dark Green Framework - Baseline Constraints)', black_style), f"%{c_data['VL']:.2f}"]
            ]
            t_col_tbl = RLTable(t_col_matrix, colWidths=[300, 232])
            t_col_tbl.setStyle(RLTableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor("#FFF3CD")), ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#DCDDE1")), ('PADDING', (0,0), (-1,-1), 4)]))
            story.append(t_col_tbl)

        doc.build(story)
        try:
            for p in img_paths.values():
                if os.path.exists(p): os.remove(p)
        except: pass
        
        # Explicit implementation: Ask the user via popup confirmation box if they want to load and open the output PDF document instantly
        open_reply = QMessageBox.question(
            self, "Open Generated Report File",
            "The multi-distance comprehensive report has been generated in PDF format successfully.\n"
            "Would you like to open the output blueprint PDF document right now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        if open_reply == QMessageBox.StandardButton.Yes:
            try:
                os.startfile(file_path)
            except AttributeError:
                import subprocess
                if sys.platform == "darwin": subprocess.call(["open", file_path])
                else: subprocess.call(["xdg-open", file_path])
            except Exception as e:
                QMessageBox.warning(self, "Execution Error", f"Could not launch PDF viewer app automatically:\n{str(e)}")

    def open_paper_addon(self):
        if not self.nash_analyzed or not self.current_report_data:
            QMessageBox.warning(self, "Warning", "Please run the simulation to generate the satisfaction layers first.")
            return
        dialog = PaperAddonDialog(self, self)
        dialog.exec()
        plt.close(dialog.figure)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec())
