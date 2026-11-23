"""Default data locations; override with environment variables or the --shape_dir / --oakbase / --mano_root flags."""
import os

SHAPE_DIR = os.environ.get("OAKINK_SHAPE_DIR", "/home/szc/data/OakInk/shape")
OAKBASE_DIR = os.environ.get("OAKBASE_DIR", "/home/szc/data/OakInk/OakBase")
MANO_ROOT = os.environ.get("MANO_ROOT", "/home/szc/work/OakInk/assets/mano_v1_2")
