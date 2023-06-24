import logging
import socket
import pickle

from pydantic import Field
from typing_extensions import Literal

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter, load_delegate


logger = logging.getLogger(__name__)

DETECTOR_KEY = "edgetpuclient"


class EdgeTpuClientConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    device: str = Field(default=None, title="Device Type")


class EdgeTpuTflClient(DetectionApi):
    type_key = DETECTOR_KEY
    HOST = 'localhost'  # The server's hostname or IP address
    PORT = 65432        # The port used by the server

    def __init__(self, detector_config: EdgeTpuClientConfig):
        self.detector_config = detector_config

    def detect_raw(self, tensor_input):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.HOST, self.PORT))
            data = pickle.dumps((self.detector_config, tensor_input))
            s.sendall(data)
            data = s.recv(1024)

        detections = pickle.loads(data)
        return detections
