from darkvessel.heads.anomaly import PiDPMAnomalyHead
from darkvessel.heads.change import ChangeHead
from darkvessel.heads.classification import ClassificationHead
from darkvessel.heads.detection import DetectionHead
from darkvessel.heads.forecast import ForecastHead
from darkvessel.heads.segmentation import SegmentationHead
from darkvessel.heads.superres import SuperResHead

__all__ = [
    "PiDPMAnomalyHead",
    "DetectionHead",
    "SegmentationHead",
    "ClassificationHead",
    "ChangeHead",
    "SuperResHead",
    "ForecastHead",
]
