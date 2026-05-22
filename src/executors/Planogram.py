"""
    Planogram comparison capsule using CLIP embeddings from ClipImage
"""

import os
import cv2
import sys
import torch
import numpy as np
from PIL import Image as PILImage
from scipy.optimize import linear_sum_assignment
from transformers import CLIPModel, CLIPProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.Planogram.src.utils.response import build_response
from capsules.Planogram.src.models.PackageModel import PackageModel


class Planogram(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.inputReferenceDetections  = self.request.get_param("inputReferenceDetections")
        self.inputTestDetections       = self.request.get_param("inputTestDetections")
        self.inputReferenceEmbeddings  = self.request.get_param("inputReferenceEmbeddings")  # ClipImage çıktısı
        self.inputTestEmbeddings       = self.request.get_param("inputTestEmbeddings")        # ClipImage çıktısı
        self.featureWeight             = self.request.get_param("featureWeight")
        self.iouWeight                 = self.request.get_param("iouWeight")
        self.treshold                  = self.request.get_param("treshold")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def run(self):
        iou_weight     = float(self.iouWeight)
        feature_weight = float(self.featureWeight)
        THRESHOLD      = float(self.treshold)

        compatibility_score = 0.0

        # ── Detection parse ──
        def parse_boxes(detections):
            boxes = []
            for d in detections:
                bb = d["boundingBox"]
                x1 = bb["left"]
                y1 = bb["top"]
                x2 = bb["left"] + bb["width"]
                y2 = bb["top"]  + bb["height"]
                boxes.append([x1, y1, x2, y2])
            return boxes

        # ── Embedding parse — ClipImage çıktısı ──
        def parse_embeddings(embeddings):
            return np.array([e["embedding"] for e in embeddings])

        ref_boxes = parse_boxes(self.inputReferenceDetections)
        curr_boxes = parse_boxes(self.inputTestDetections)

        n_ref  = len(ref_boxes)
        n_curr = len(curr_boxes)

        if n_ref == 0 or n_curr == 0:
            self.request.data["compatibilityScore"] = compatibility_score
            self.data = str(compatibility_score)
            return build_response(context=self)

        ref_embs  = parse_embeddings(self.inputReferenceEmbeddings)
        curr_embs = parse_embeddings(self.inputTestEmbeddings)

        # ── IoU matrisi ──
        def compute_iou(b1, b2):
            x1 = max(b1[0], b2[0]);  y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2]);  y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
            a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = a1 + a2 - inter
            return inter / union if union > 0 else 0.0

        iou_mat = np.zeros((n_ref, n_curr))
        for i, rb in enumerate(ref_boxes):
            for j, cb in enumerate(curr_boxes):
                iou_mat[i, j] = compute_iou(rb, cb)

        # ── Feature matrisi ──
        # Embedding'ler normalize edilmemişse normalize et
        ref_norms  = np.linalg.norm(ref_embs,  axis=1, keepdims=True)
        curr_norms = np.linalg.norm(curr_embs, axis=1, keepdims=True)
        ref_embs   = ref_embs  / np.where(ref_norms  > 0, ref_norms,  1)
        curr_embs  = curr_embs / np.where(curr_norms > 0, curr_norms, 1)

        feature_mat = (ref_embs @ curr_embs.T + 1) / 2

        # ── Birleştir + Hungarian ──
        combined         = iou_weight * iou_mat + feature_weight * feature_mat
        row_ind, col_ind = linear_sum_assignment(1 - combined)

        matched_count = sum(
            1 for r, c in zip(row_ind, col_ind)
            if combined[r, c] >= THRESHOLD
        )

        compatibility_score = round(matched_count / n_ref * 100, 2)

        self.request.data["compatibilityScore"] = compatibility_score
        self.data = str(compatibility_score)

        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()