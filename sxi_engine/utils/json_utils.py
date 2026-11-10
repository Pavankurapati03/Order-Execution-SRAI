import math
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

def make_json_safe(obj):
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None

    if isinstance(obj, pd.DataFrame):
        return make_json_safe(obj.to_dict(orient="records"))

    if isinstance(obj, pd.Series):
        return make_json_safe(obj.to_dict())

    if isinstance(obj, np.ndarray):
        return make_json_safe(obj.tolist())

    if isinstance(obj, np.generic):
        return make_json_safe(obj.item())

    if isinstance(obj, Decimal):
        return float(obj) if math.isfinite(float(obj)) else None

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")

    return str(obj)
