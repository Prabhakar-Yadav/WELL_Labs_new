"""
Common utilities for Well-LAB-ALL- pipeline scripts.
Provides path resolution and shared helpers.
"""
import os
import sys
from pathlib import Path

def get_project_root():
    """Get project root directory (3 levels up from this file)."""
    return Path(__file__).parent.parent.parent.resolve()

def get_data_dir():
    """Get data directory path."""
    return get_project_root() / "data"

def get_config_dir():
    """Get config directory path."""
    return get_project_root() / "config"

def get_models_dir():
    """Get models directory path."""
    return get_config_dir() / "models"

def get_outputs_dir():
    """Get outputs directory path."""
    return get_project_root() / "outputs"

def get_output_subdir(subdir):
    """Get specific output subdirectory."""
    path = get_outputs_dir() / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_raw_ortho_path(filename):
    """Get path to raw drone orthomosaic."""
    return get_data_dir() / "raw_drone_orthos" / filename

def get_alignment_path(filename):
    """Get path to alignment shapefile."""
    return get_data_dir() / "alignments" / filename

def get_training_labels_path(subdir=""):
    """Get path to training labels directory."""
    base = get_data_dir() / "training_labels"
    if subdir:
        return base / subdir
    return base

def ensure_dir(path):
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return Path(path)

# Classification colors (BGR for OpenCV)
COLORS = {
    'clear': (0, 190, 0),
    'partial': (235, 205, 0),
    'moderate': (255, 140, 0),
    'blocked': (220, 20, 20),
    'full': (220, 20, 20),
}

# Status ranking for evaluation
RANK = {'clear': 0, 'partial': 1, 'blocked': 2, 'moderate': 1}

def classify_veg(veg_frac, water_frac=None):
    """Classify blockage from vegetation fraction."""
    if water_frac is not None and veg_frac > 0.55 and water_frac < 0.20:
        return 'blocked'
    if veg_frac > 0.55:
        return 'blocked'
    if veg_frac > 0.35:
        return 'partial'
    return 'clear'

def to_4class(c3, pct):
    """Convert 3-class to 4-class with percentage threshold."""
    if c3 == 'clear':
        return 'clear'
    if c3 == 'blocked':
        return 'blocked'
    return 'partial' if pct <= 30 else 'moderate'
