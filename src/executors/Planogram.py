"""
    Planogram comparison component using CLIP for feature matching
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
        self.inputTestDetections      = self.request.get_param("inputTestDetections")
        self.featureWeight            = self.request.get_param("featureWeight")
        self.iouWeight                = self.request.get_param("iouWeight")
        self.inputImageOne            = self.request.get_param("inputImageOne")
        self.inputImageTwo            = self.request.get_param("inputImageTwo")
        print(self.inputImageOne)
        print(self.inputImageTwo)
        self.clip_model = bootstrap["clip_model"]
        self.processor  = bootstrap["processor"]
        self.device     = bootstrap["device"]

    @staticmethod
    def bootstrap(config: dict) -> dict:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        device     = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model = clip_model.to(device).eval()

        return {
            "clip_model": clip_model,
            "processor":  processor,
            "device":     device,
        }

    def run(self):
        import torch
        import numpy as np
        from PIL import Image as PILImage
        from scipy.optimize import linear_sum_assignment

        iou_weight     = float(self.iouWeight)
        feature_weight = float(self.featureWeight)
        THRESHOLD      = 0.20

        # ── Görüntüleri Redis'ten oku ──
        ref_frame  = Image.get_frame(img=self.inputImageOne, redis_db=self.redis_db)
        curr_frame = Image.get_frame(img=self.inputImageTwo, redis_db=self.redis_db)

        ref_np  = np.asarray(ref_frame.value).astype(np.uint8)
        curr_np = np.asarray(curr_frame.value).astype(np.uint8)

        ref_pil  = PILImage.fromarray(ref_np[..., ::-1])
        curr_pil = PILImage.fromarray(curr_np[..., ::-1])

        # ── Detection parse — tüm sınıflar ──
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

        ref_boxes  = parse_boxes(self.inputReferenceDetections)
        curr_boxes = parse_boxes(self.inputTestDetections)

        n_ref  = len(ref_boxes)
        n_curr = len(curr_boxes)

        print(f"n_ref: {n_ref}  n_curr: {n_curr}")
        print("REF BOXES:", ref_boxes[:3])
        print("CURR BOXES:", curr_boxes[:3])
        print("REF PIL SIZE:", ref_pil.size)

        if n_ref == 0 or n_curr == 0:
            self.request.data["compatibilityScore"] = 0.0
            self.data = "0.0"
            return build_response(context=self)

        # ── Batch CLIP embedding ──
        def get_embeddings(pil_image, boxes):
            crops = []
            for box in boxes:
                x1 = max(0, int(box[0]));  y1 = max(0, int(box[1]))
                x2 = min(pil_image.width,  int(box[2]))
                y2 = min(pil_image.height, int(box[3]))
                if x2 <= x1 or y2 <= y1:
                    crops.append(PILImage.new("RGB", (224, 224)))
                    continue
                crops.append(pil_image.crop((x1, y1, x2, y2)))

            inputs = self.processor(images=crops, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.clip_model.get_image_features(**inputs)
                embs    = outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
                embs    = embs / embs.norm(p=2, dim=-1, keepdim=True)

            return embs.cpu().numpy()

        ref_embs  = get_embeddings(ref_pil,  ref_boxes)
        curr_embs = get_embeddings(curr_pil, curr_boxes)

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
        feature_mat = (ref_embs @ curr_embs.T + 1) / 2

        # ── Birleştir + Hungarian ──
        combined         = iou_weight * iou_mat + feature_weight * feature_mat
        row_ind, col_ind = linear_sum_assignment(1 - combined)

        # ── Skorları yazdır ──
        print("=== COMBINED SKORLAR ===")
        for r, c in zip(row_ind, col_ind):
            print(f"  [{r}↔{c}]  iou:{iou_mat[r,c]:.3f}  feat:{feature_mat[r,c]:.3f}  combined:{combined[r,c]:.3f}")

        matched_count = sum(
            1 for r, c in zip(row_ind, col_ind)
            if combined[r, c] >= THRESHOLD
        )

        compatibility_score = round(matched_count / n_ref * 100, 2)
        print(f"=== SONUÇ: {compatibility_score} ===")

        self.request.data["compatibilityScore"] = compatibility_score
        self.data = str(compatibility_score)

        return build_response(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()