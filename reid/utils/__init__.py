from pathlib import Path
from typing import Union, Optional

FILE = Path(__file__).resolve() # reid/utils
ROOT = FILE.parents[1]  # reid
ASSETS = ROOT / "assets"  # default images
DEFAULT_CFG_ROOT = ROOT / "cfg"
DEFAULT_CFG_PATH = DEFAULT_CFG_ROOT / "default.yaml"

def get_cfg_path(path: Union[str, Path, None] = None) -> Optional[Path]:
    """
    Resolve a configuration file path.

    - return path if path exists
    - track file recursively from DEFAULT_CFG_ROOT
    - return default.yaml if path is config.yaml and does not exist
    - return None if path does not exist
    """
    path = path if path else Path(ROOT.parents[0] / "config.yaml")
    path = Path(path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".yaml")
    if path.is_file():
        return path.resolve()

    for f in DEFAULT_CFG_ROOT.rglob(path.name) :
        if f.is_file(): return f.resolve()

    if path.name == "config.yaml": return DEFAULT_CFG_PATH
    return None

if __name__ == "__main__":
    import os
    os.chdir("/home/jhj/project_ws/lumipet_ws/re-id_test/reid/cfg/")
    print(f"result = {get_cfg_path()}")