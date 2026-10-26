"""Care-label symbol recognition: photo -> care symbols -> structured result.

    from care_label_recognition import recognize_care_label
    result = recognize_care_label("label.jpg")
"""
from .recognizer import CareLabelRecognizer, recognize_care_label

__all__ = ["recognize_care_label", "CareLabelRecognizer"]
__version__ = "1.3.1"
