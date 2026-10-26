"""Where the engagement code finds external data and model code. Defaults follow the author's layout under ~/Code
(see ~/Code/shared/README.md); each path can be overridden with an environment variable."""
import os
from pathlib import Path

CODE = Path(os.environ.get('ENGAGE3D_CODE', Path.home()/'Code'))
H4D = Path(os.environ.get('ENGAGE3D_H4D', CODE/'kakashi-research/external/harmony4d'))           # Harmony4D tree
H4D_ZIPS = Path(os.environ.get('ENGAGE3D_H4D_ZIPS', CODE/'shared/datasets/harmony4d/test-split-full/test'))
VICOS = Path(os.environ.get('ENGAGE3D_VICOS', CODE/'movement-research/data/bjj_positions_vicos'))  # ViCoS BJJ
PHMR = Path(os.environ.get('ENGAGE3D_PHMR', CODE/'movement-research/repos/PromptHMR'))             # PromptHMR repo
