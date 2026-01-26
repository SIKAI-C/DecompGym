from typing import Any, Dict, Optional
from pathlib import Path
import json

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def serialize_iter_records(records: list[Any]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for rec in records:
        serialized.append(
            {
                "iteration": rec.iteration,
                "elapsed_time": rec.elapsed_time,
                "iter_time": rec.iter_time,
                "master_obj": rec.master_obj,
                "sub_bound": rec.sub_bound,
                "global_lb": rec.global_lb,
                "global_ub": rec.global_ub,
                "n_feas_cuts": rec.n_feas_cuts,
                "n_opt_cuts": rec.n_opt_cuts,
            }
        )
    return serialized


def save_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None