# Copyright (c) Facebook, Inc. and its affiliates.
import os
from functools import lru_cache
import torch
import numpy as np
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from vlpart.config import add_vlpart_config
from vlpart.modeling.text_encoder.text_encoder import build_text_encoder

import supervision as sv

code_path = os.path.abspath(os.path.join(__file__, '../..'))
ROOT=lambda *f: os.path.join(code_path, *f)

BUILDIN_CLASSIFIER = {
    'pascal_part': 'datasets/metadata/pascal_part_clip_RN50_a+cname.npy',
    'partimagenet': 'datasets/metadata/partimagenet_clip_RN50_a+cname.npy',
    'paco': 'datasets/metadata/paco_clip_RN50_a+cname.npy',
    'lvis': 'datasets/metadata/lvis_v1_clip_RN50_a+cname.npy',
    'coco': 'datasets/metadata/coco_clip_RN50_a+cname.npy',
    'voc': 'datasets/metadata/voc_clip_RN50_a+cname.npy',
}

BUILDIN_METADATA_PATH = {
    'pascal_part': 'pascal_part_val',
    'partimagenet': 'partimagenet_val',
    'paco': 'paco_lvis_v1_val',
    'lvis': 'lvis_v1_val',
    'coco': 'coco_2017_val',
    'voc': 'voc_2007_val',
}


# from: https://github.com/beasteers/VLPart/blob/main/MODEL_ZOO.md
BASE_URL = "https://github.com/PeizeSun/VLPart/releases/download/v0.1/"
CONFIGS = {
    # Cross-dataset part segmentation on PartImageNet
    "r50_pascal_part": ("configs/pascal_part/r50_pascalpart.yaml","r50_pascalpart.pth"),
    "r50_pascalpart_ins11": ("configs/partimagenet_ablation/r50_pascalpart_ins11.yaml","r50_pascalpart_ins11.pth"),
    "r50_pascalpart_ins11_ins11parsed": ("configs/partimagenet_ablation/r50_pascalpart_ins11_ins11parsed.yaml","r50_pascalpart_ins11_ins11parsed.pth"),
    "r50_lvis_paco_pascalpart": ("configs/partimagenet_ablation/r50_lvis_paco_pascalpart.yaml","r50_lvis_paco_pascalpart.pth"),
    "r50_lvis_paco_pascalpart_ins11": ("configs/partimagenet_ablation/r50_lvis_paco_pascalpart_ins11.yaml","r50_lvis_paco_pascalpart_ins11.pth"),
    "r50_lvis_paco_pascalpart_ins11_ins11parsed": ("configs/partimagenet_ablation/r50_lvis_paco_pascalpart_ins11_ins11parsed.yaml","r50_lvis_paco_pascalpart_ins11_ins11parsed.pth"),
    
    # Cross-category part segmentation within Pascal Part
    "r50_pascalpartbase": ("configs/pascal_part_ablation/r50_pascalpartbase.yaml","r50_pascalpartbase.pth"),
    "r50_pascalpartbase_voc": ("configs/pascal_part_ablation/r50_pascalpartbase_voc.yaml","r50_pascalpartbase_voc.pth"),
    "r50_pascalpartbase_voc_ins20": ("configs/pascal_part_ablation/r50_pascalpartbase_voc_ins20.yaml","r50_pascalpartbase_voc_ins20.pth"),
    "r50_pascalpartbase_voc_ins20_ins20parsed": ("configs/pascal_part_ablation/r50_pascalpartbase_voc_ins20_ins20parsed.yaml","r50_pascalpartbase_voc_ins20_ins20parsed.pth"),
    
    # Open-vocabulary object detection and part segmentation

    # dataset-specific
    "r50_voc": ("configs/voc/r50_voc.yaml", "r50_voc.pth"),
    "r50_coco": ("configs/voc/r50_coco.yaml", "r50_coco.pth"),
    "r50_lvis": ("configs/voc/r50_lvis.yaml", "r50_lvis.pth"),
    "r50_partimagenet": ("configs/voc/r50_partimagenet.yaml", "r50_partimagenet.pth"),
    "r50_pascalpart": ("configs/voc/r50_pascalpart.yaml", "r50_pascalpart.pth"),
    "r50_paco": ("configs/voc/r50_paco.yaml", "r50_paco.pth"),

    # joint
    "r50_lvis_paco": ("configs/joint/r50_lvis_paco.yaml", "r50_lvis_paco.pth"),
    "r50_lvis_paco_pascalpart": ("configs/joint/r50_lvis_paco_pascalpart.yaml", "r50_lvis_paco_pascalpart.pth"),
    "r50_lvis_paco_pascalpart_partimagenet": ("configs/joint/r50_lvis_paco_pascalpart_partimagenet.yaml", "r50_lvis_paco_pascalpart_partimagenet.pth"),
    "r50_lvis_paco_pascalpart_partimagenet_in": ("configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_in.yaml", "r50_lvis_paco_pascalpart_partimagenet_in.pth"),
    "r50_lvis_paco_pascalpart_partimagenet_inparsed": ("configs/joint_in/r50_lvis_paco_pascalpart_partimagenet_inparsed.yaml", "r50_lvis_paco_pascalpart_partimagenet_inparsed.pth"),

    # Open-vocabulary object detection and part segmentation

    # dataset-specific
    "swinbase_cascade_voc": ("configs/voc/swinbase_cascade_voc.yaml", "swinbase_cascade_voc.pth"),
    "swinbase_cascade_coco": ("configs/voc/swinbase_cascade_coco.yaml", "swinbase_cascade_coco.pth"),
    "swinbase_cascade_lvis": ("configs/voc/swinbase_cascade_lvis.yaml", "swinbase_cascade_lvis.pth"),
    "swinbase_cascade_partimagenet": ("configs/voc/swinbase_cascade_partimagenet.yaml", "swinbase_cascade_partimagenet.pth"),
    "swinbase_cascade_pascalpart": ("configs/voc/swinbase_cascade_pascalpart.yaml", "swinbase_cascade_pascalpart.pth"),
    "swinbase_cascade_paco": ("configs/voc/swinbase_cascade_paco.yaml", "swinbase_cascade_paco.pth"),

    # joint
    "swinbase_cascade_lvis_paco": ("configs/joint/swinbase_cascade_lvis_paco.yaml", "swinbase_cascade_lvis_paco.pth"),
    "swinbase_cascade_lvis_paco_pascalpart": ("configs/joint/swinbase_cascade_lvis_paco_pascalpart.yaml", "swinbase_cascade_lvis_paco_pascalpart.pth"),
    "swinbase_cascade_lvis_paco_pascalpart_partimagenet": ("configs/joint/swinbase_cascade_lvis_paco_pascalpart_partimagenet.yaml", "swinbase_cascade_lvis_paco_pascalpart_partimagenet.pth"),
    "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in": ("configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.yaml", "swinbase_cascade_lvis_paco_pascalpart_partimagenet_in.pth"),
    "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed": ("configs/joint_in/swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.yaml", "swinbase_cascade_lvis_paco_pascalpart_partimagenet_inparsed.pth"),
}



box_annotator = sv.BoxAnnotator()
mask_annotator = sv.MaskAnnotator()

def draw(image, outputs, model):
    detections = sv.Detections(
        xyxy=outputs["instances"].pred_boxes.tensor.cpu().numpy(),
        mask=outputs["instances"].pred_masks.cpu().numpy() if hasattr(outputs["instances"], 'pred_masks') else None,
        confidence=outputs["instances"].scores.cpu().numpy(),
        class_id=outputs["instances"].pred_classes.cpu().int().numpy(),
    )
    image = mask_annotator.annotate(image, detections)
    image = box_annotator.annotate(image, detections, labels=model.labels[detections.class_id.astype(int)])
    return image

class VLPart(torch.nn.Module):
    # def __init__(self, vocab=None, conf_threshold=0.5, box_conf_threshold=0.5, masks=False, one_class_per_proposal=True, patch_for_embeddings=True, prompt=DEFAULT_PROMPT, device=device):
    def __init__(self, vocab='lvis+paco', confidence_threshold=0.5, config_name="swinbase_cascade_lvis_paco"):
        super().__init__()
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_vlpart_config(cfg)
        config_file, checkpoint = CONFIGS[config_name]
        cfg.merge_from_file(ROOT(config_file))
        cfg.SRC_ROOT = code_path
        cfg.MODEL.WEIGHTS = BASE_URL + checkpoint
        # Set score_threshold for builtin models
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        self.predictor = DefaultPredictor(cfg)
        self.text_encoder = build_text_encoder(pretrain=True)
        self.text_encoder.eval()
        if vocab:
            self.set_vocab(vocab)

    def set_vocab(self, text):
        if isinstance(text, (list, set)):
            text = tuple(text)
        self.metadata, classifier = get_vocabulary(text, self.text_encoder)
        self.labels = np.asarray(self.metadata.thing_classes)
        set_zs_weight(self.predictor.model, classifier)
        
    def forward(self, img):
        return self.predictor(img)
    

def get_clip_embeddings(vocabulary, text_encoder, prompt='a '):
    return text_encoder([
        prompt + x.lower().replace(':', ' ') 
        for x in vocabulary
    ]).detach().permute(1, 0).contiguous().cpu()


@lru_cache(8)
def get_vocabulary(vocabulary, text_encoder, classifier=None, metadata_name=None):
    # built-in vocabulary
    if isinstance(vocabulary, str):
        if vocabulary in BUILDIN_METADATA_PATH:
            metadata_name = BUILDIN_METADATA_PATH[vocabulary]
            classifier = BUILDIN_CLASSIFIER[vocabulary]
        elif ',' in vocabulary:
            vocabulary = [v.strip() for v in vocabulary.split(',')]
        else:
            metadata_name = vocabulary
            vocabulary = sorted({
                c for name in vocabulary.split('+')
                for cs in MetadataCatalog.get(BUILDIN_METADATA_PATH[name.strip()]).thing_classes
                for c in cs
            })
    
    # load classifier from file
    if isinstance(classifier, str):
        if classifier.endswith('npy'):
            classifier = np.load(F(classifier))
            classifier = torch.tensor(
                classifier, dtype=torch.float32).permute(1, 0).contiguous()  # dim x C
        elif classifier.endswith('pth'):
            classifier = torch.load(F(classifier), map_location='cpu')
            classifier = classifier.clone().detach().permute(1, 0).contiguous()  # dim x C
        else:
            raise NotImplementedError
    
    metadata_name = metadata_name or '__unused'
    metadata = MetadataCatalog.get(metadata_name)

    # custom vocabulary - get embeddings
    if classifier is None:
        if metadata_name in MetadataCatalog:
            MetadataCatalog.remove(metadata_name)
            metadata = MetadataCatalog.get(metadata_name)
        metadata.thing_classes = vocabulary
        classifier = get_clip_embeddings(metadata.thing_classes, text_encoder)
    return metadata, classifier


def set_zs_weight(model, zs_weight):
    zs_weight = torch.cat(
        [zs_weight, zs_weight.new_zeros((zs_weight.shape[0], 1))],
        dim=1) # D x (C + 1)
    zs_weight = F.normalize(zs_weight, p=2, dim=0)
    zs_weight = zs_weight.to(model.device)

    ps = model.roi_heads.box_predictor
    for p in ps if isinstance(ps, torch.nn.ModuleList) else [ps]:
        p.cls_score.zs_weight_inference = zs_weight
