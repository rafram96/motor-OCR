import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from segmentation import config as seg_config


def test_line_thresholds_are_valid():
    assert seg_config.MIN_LINEAS_SEPARADORA >= 0
    assert seg_config.MAX_LINEAS_SEPARADORA >= seg_config.MIN_LINEAS_SEPARADORA


def test_fuzzy_and_catalog_are_configured():
    assert seg_config.FUZZY_SCORE_MINIMO > 0
    assert seg_config.FUZZY_SCORE_MINIMO <= 100
    assert isinstance(seg_config.CARGOS_BASE, list)
    assert len(seg_config.CARGOS_BASE) > 0
    assert isinstance(seg_config.NORMALIZACIONES, dict)