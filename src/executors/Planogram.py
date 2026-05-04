"""
    It is one of the preprocessing components in which the image is rotated.
"""

import os
import cv2
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.component import Component
from sdks.novavision.src.helper.executor import Executor
from components.Planogram.src.utils.response import build_response
from components.Planogram.src.models.PackageModel import PackageModel


class Planogram(Component):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.inputReferenceDetections = self.request.get_param("inputReferenceDetections")
        self.inputTestDetections = self.request.get_param("inputTestDetections")
        self.featureWeight = self.request.get_param("featureWeight")
        self.iouWeight = self.request.get_param("iouWeight")
        self.inputImageOne = self.request.get_param("inputImageOne")
        self.inputImageTwo = self.request.get_param("inputImageTwo")
        print(self.featureWeight)
        print(self.iouWeight)

        print("merhaba")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        self.image = self.inputImageOne
        self.outputData = self.featureWeight

        packageModel = build_response(context=self)
        return packageModel


if "__main__" == __name__:
    Executor(sys.argv[1]).run()