# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_coco_json, load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc
from .ACDC import register_ACDC_instances
from .shift import register_shift_instances

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}


def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

def register_cityscapes_c(root):
    """
    Register Cityscapes-C in three views:

      <corruption>         : detection-only COCO dataset (legacy name)
      <corruption>_semseg  : semantic-segmentation dataset
      <corruption>_mtl     : detection + semantic segmentation

    Cityscapes-C images preserve the original Cityscapes filenames, therefore
    the clean Cityscapes val labelTrainIds are also the correct semantic GT.
    """
    corruption_types = [
        "gaussian_noise", "shot_noise", "impulse_noise",
        "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
        "snow", "frost", "fog", "brightness", "contrast",
        "elastic_transform", "pixelate", "jpeg_compression",
    ]

    cityscapes_root = os.path.join(root, "cityscapes")
    gt_root = os.path.join(cityscapes_root, "gtFine", "val")
    json_file = os.path.join(
        cityscapes_root,
        "annotations",
        "instancesonly_filtered_gtFine_val.json",
    )

    cityscapes_meta = _get_builtin_metadata("cityscapes")

    for corrupt_type in corruption_types:
        corrupt_root = os.path.join(root, corrupt_type)
        image_root = os.path.join(corrupt_root, "leftImg8bit", "val")

        # ------------------------------------------------------------------
        # 1. Legacy detection-only registration.
        # Keep this exact name/behaviour for all previous experiments.
        # ------------------------------------------------------------------
        register_coco_instances(
            corrupt_type,
            {},
            json_file,
            corrupt_root,
        )

        # ------------------------------------------------------------------
        # 2. Semantic-segmentation view.
        # ------------------------------------------------------------------
        semseg_name = f"{corrupt_type}_semseg"

        DatasetCatalog.register(
            semseg_name,
            lambda image_root=image_root, gt_root=gt_root:
                load_cityscapes_semantic(image_root, gt_root),
        )

        MetadataCatalog.get(semseg_name).set(
            image_dir=image_root,
            gt_dir=gt_root,
            image_root=image_root,
            sem_seg_root=gt_root,
            evaluator_type="sem_seg",
            ignore_label=255,
            stuff_classes=list(cityscapes_meta["stuff_classes"]),
        )

        # ------------------------------------------------------------------
        # 3. Multi-task view: COCO bbox records + Cityscapes semantic GT.
        # ------------------------------------------------------------------
        mtl_name = f"{corrupt_type}_mtl"

        def _load_cityscapes_c_mtl(
            json_file=json_file,
            corrupt_root=corrupt_root,
            gt_root=gt_root,
            dataset_name=mtl_name,
        ):
            dicts = load_coco_json(
                json_file,
                corrupt_root,
                dataset_name=dataset_name,
            )

            missing = []
            for d in dicts:
                basename = os.path.basename(d["file_name"])
                suffix = "_leftImg8bit.png"

                if not basename.endswith(suffix):
                    missing.append(d["file_name"])
                    continue

                stem = basename[:-len(suffix)]
                city = stem.split("_", 1)[0]

                sem_seg_file = os.path.join(
                    gt_root,
                    city,
                    f"{stem}_gtFine_labelTrainIds.png",
                )

                if not os.path.isfile(sem_seg_file):
                    missing.append(sem_seg_file)
                    continue

                d["sem_seg_file_name"] = sem_seg_file

            if missing:
                preview = "\n".join(str(x) for x in missing[:5])
                raise FileNotFoundError(
                    f"{dataset_name}: missing semantic GT for "
                    f"{len(missing)} records. First entries:\n{preview}"
                )

            return dicts

        DatasetCatalog.register(mtl_name, _load_cityscapes_c_mtl)

        MetadataCatalog.get(mtl_name).set(
            thing_classes=list(cityscapes_meta["thing_classes"]),
            stuff_classes=list(cityscapes_meta["stuff_classes"]),
            json_file=json_file,
            image_root=corrupt_root,
            sem_seg_root=gt_root,
            evaluator_type="coco_sem_seg",
            ignore_label=255,
        )


def register_ACDC(root):
    register_ACDC_instances("acdc_fog", root+"/ACDC/gt_detection/fog/instancesonly_fog_train_gt_detection.json", root+"/ACDC/rgb_anon")
    register_ACDC_instances("acdc_night", root+"/ACDC/gt_detection/night/instancesonly_night_train_gt_detection.json", root+"/ACDC/rgb_anon")
    register_ACDC_instances("acdc_rain", root+"/ACDC/gt_detection/rain/instancesonly_rain_train_gt_detection.json", root+"/ACDC/rgb_anon")
    register_ACDC_instances("acdc_snow", root+"/ACDC/gt_detection/snow/instancesonly_snow_train_gt_detection.json", root+"/ACDC/rgb_anon")


def register_foggy_cityscapes(root):
    """Foggy Cityscapes val set (beta=0.02) for Cityscapes→FoggyCityscapes eval.

    Also registers foggy_cityscapes_val_mtl which adds sem_seg_file_name from
    Cityscapes gtFine/val (same GT labels, foggy images) so both bbox AP and
    mIoU can be evaluated with evaluator_type='coco_sem_seg'.
    """
    from .ACDC import register_ACDC_instances as _reg, load_ACDC_json
    from .ACDC import dataset_id_to_contiguous_id as _acdc_id_map, CLASS_NAMES as _acdc_classes
    json_file = os.path.join(root, "cityscapes_foggy/annotations/instancesonly_filtered_gtFine_val_foggy_beta_0.02.json")
    image_root = os.path.join(root, "cityscapes_foggy/leftImg8bit_foggyDBF/val")
    cs_gt_root = os.path.join(root, "cityscapes/gtFine/val")

    if not (os.path.isfile(json_file) and os.path.isdir(image_root)):
        print(f"[builtin] foggy_cityscapes_val not registered (files not found under {root})")
        return

    _reg("foggy_cityscapes_val", json_file, image_root)

    # MTL variant: det bbox annotations + semantic seg GT from Cityscapes gtFine.
    # Image stem: {city}_{frame}_{idx}_leftImg8bit_foggy_beta_0.02.png
    # GT label:   {cs_gt_root}/{city}/{city}_{frame}_{idx}_gtFine_labelTrainIds.png
    if os.path.isdir(cs_gt_root):
        cityscapes_meta = _get_builtin_metadata("cityscapes")

        def _load_foggy_mtl(json_file=json_file, image_root=image_root, cs_gt_root=cs_gt_root):
            dicts = load_ACDC_json(json_file, image_root, "foggy_cityscapes_val_mtl")
            for d in dicts:
                basename = os.path.basename(d["file_name"])
                stem = basename.replace("_leftImg8bit_foggy_beta_0.02.png", "")
                city = stem.split("_")[0]
                label_path = os.path.join(cs_gt_root, city, f"{stem}_gtFine_labelTrainIds.png")
                if os.path.isfile(label_path):
                    d["sem_seg_file_name"] = label_path
            return dicts

        DatasetCatalog.register("foggy_cityscapes_val_mtl", _load_foggy_mtl)
        MetadataCatalog.get("foggy_cityscapes_val_mtl").set(
            thing_classes=list(_acdc_classes),
            stuff_classes=cityscapes_meta["stuff_classes"],
            thing_dataset_id_to_contiguous_id=dict(_acdc_id_map),
            json_file=json_file,
            image_root=image_root,
            sem_seg_root=cs_gt_root,
            evaluator_type="coco_sem_seg",
            ignore_label=255,
        )
    else:
        print(f"[builtin] foggy_cityscapes_val_mtl not registered (Cityscapes gtFine not found at {cs_gt_root})")

    # cityscapes_val_mtl: clean Cityscapes val for loopback / forgetting eval.
    # Reuses the foggy MTL machinery but points at clear-weather val images
    # and their own annotations (bbox JSON is the same instancesonly_filtered
    # file Cityscapes ships in annotations/).
    cs_val_json = os.path.join(root, "cityscapes/annotations/instancesonly_filtered_gtFine_val.json")
    cs_val_image_root = os.path.join(root, "cityscapes/leftImg8bit/val")
    if os.path.isfile(cs_val_json) and os.path.isdir(cs_val_image_root) and os.path.isdir(cs_gt_root):
        def _load_cs_val_mtl(json_file=cs_val_json, image_root=cs_val_image_root, cs_gt_root=cs_gt_root):
            dicts = load_ACDC_json(json_file, image_root, "cityscapes_val_mtl")
            for d in dicts:
                basename = os.path.basename(d["file_name"])
                stem = basename.replace("_leftImg8bit.png", "")
                city = stem.split("_")[0]
                label_path = os.path.join(cs_gt_root, city, f"{stem}_gtFine_labelTrainIds.png")
                if os.path.isfile(label_path):
                    d["sem_seg_file_name"] = label_path
            return dicts

        DatasetCatalog.register("cityscapes_val_mtl", _load_cs_val_mtl)
        MetadataCatalog.get("cityscapes_val_mtl").set(
            thing_classes=list(_acdc_classes),
            stuff_classes=cityscapes_meta["stuff_classes"],
            thing_dataset_id_to_contiguous_id=dict(_acdc_id_map),
            json_file=cs_val_json,
            image_root=cs_val_image_root,
            sem_seg_root=cs_gt_root,
            evaluator_type="coco_sem_seg",
            ignore_label=255,
        )
    else:
        print(f"[builtin] cityscapes_val_mtl not registered (missing {cs_val_json} or {cs_val_image_root})")
    
    
def register_shift(root):
    register_shift_instances("shift_train", root+"/shift/annotations/gtfine_clear_train.json", root+"/shift/rgb_anon/images/train/front")
    register_shift_instances("shift_test", root+"/shift/annotations/gtfine_clear_val.json", root+"/shift/rgb_anon/images/val/front")
    register_shift_instances("shift_cloudy", root+"/shift/annotations/gtfine_cloudy_val.json", root+"/shift/rgb_anon/images/val/front")
    register_shift_instances("shift_overcast", root+"/shift/annotations/gtfine_overcast_val.json", root+"/shift/rgb_anon/images/val/front")
    register_shift_instances("shift_rainy", root+"/shift/annotations/gtfine_rainy_val.json", root+"/shift/rgb_anon/images/val/front")
    register_shift_instances("shift_foggy", root+"/shift/annotations/gtfine_foggy_val.json", root+"/shift/rgb_anon/images/val/front")
    
# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "~/AMROD/datasets"))
    if not os.path.exists(_root):
        print(f"Dataset dir \" {_root} \" does not exist")
        raise 
    register_cityscapes_c(_root)
    register_ACDC(_root)
    register_shift(_root)
    try:
        register_foggy_cityscapes(_root)
    except Exception as e:
        print(f"[builtin] foggy_cityscapes_val not registered: {e}")
    # MTL-CTTA extension: register standard Cityscapes panoptic datasets and a
    # merged instance+sem_seg variant ("cityscapes_fine_mtl_*") suitable for
    # Panoptic-FPN-style joint det+seg training.
    try:
        register_all_cityscapes(_root)
    except Exception as e:
        print(f"[builtin] cityscapes_fine_{{instance_seg,sem_seg}} not registered: {e}")
    try:
        register_all_cityscapes_panoptic(_root)
    except Exception as e:
        print(f"[builtin] cityscapes_fine_panoptic not registered: {e}")

    def _register_cityscapes_mtl(root):
        # Merges per-image instance dicts (bbox/mask/annotations) with the
        # sem_seg_file_name from the semantic-seg loader so a single dataset
        # yields both instance annotations AND semantic segmentation targets.
        for split in ("train", "val"):
            name = f"cityscapes_fine_mtl_{split}"
            image_dir = os.path.join(root, f"cityscapes/leftImg8bit/{split}")
            gt_dir = os.path.join(root, f"cityscapes/gtFine/{split}")
            meta = _get_builtin_metadata("cityscapes")

            def _load(image_dir=image_dir, gt_dir=gt_dir):
                inst = load_cityscapes_instances(
                    image_dir, gt_dir, from_json=True, to_polygons=True
                )
                sem = load_cityscapes_semantic(image_dir, gt_dir)
                sem_by_file = {r["file_name"]: r["sem_seg_file_name"] for r in sem}
                for d in inst:
                    if d["file_name"] in sem_by_file:
                        d["sem_seg_file_name"] = sem_by_file[d["file_name"]]
                return inst

            DatasetCatalog.register(name, _load)
            MetadataCatalog.get(name).set(
                image_dir=image_dir,
                gt_dir=gt_dir,
                evaluator_type="cityscapes_instance",
                ignore_label=255,
                **meta,
            )

    try:
        _register_cityscapes_mtl(_root)
    except Exception as e:
        print(f"[builtin] cityscapes_fine_mtl not registered: {e}")

    # MTL-CTTA extension: COCO-format bbox dataset for standard COCO bbox AP
    # evaluation (matches AMROD's reporting convention). JSON must be
    # generated once via detectron2.data.datasets.coco.convert_to_coco_json.
    bbox_json = os.path.join(_root, "annotations/cityscapes_bbox_val.json")
    if os.path.exists(bbox_json):
        try:
            register_coco_instances(
                "cityscapes_bbox_val",
                {},
                bbox_json,
                os.path.join(_root, "cityscapes/leftImg8bit/val"),
            )
        except Exception as e:
            print(f"[builtin] cityscapes_bbox_val not registered: {e}")

    # MTL-CTTA extension: ACDC semseg-only and MTL variants. The base
    # register_ACDC() above already registers det-only variants
    # ("acdc_fog", "acdc_night", "acdc_rain", "acdc_snow"). Here we add:
    #   acdc_{weather}_semseg -> sem_seg only    (for CoTTA_SemSeg eval)
    #   acdc_{weather}_mtl    -> det + sem_seg   (for CTCMT_MTL eval)
    def _register_acdc_semseg_and_mtl(root):
        from .ACDC import load_ACDC_json, CLASS_NAMES as _ACDC_CLASS_NAMES
        acdc_root = os.path.join(root, "ACDC")
        if not os.path.isdir(acdc_root):
            return
        weathers = ("fog", "night", "rain", "snow")
        cityscapes_meta = {
            "stuff_classes": [
                "road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain",
                "sky", "person", "rider", "car", "truck", "bus",
                "train", "motorcycle", "bicycle",
            ],
        }
        thing_classes = list(_ACDC_CLASS_NAMES)

        image_root_base = os.path.join(acdc_root, "rgb_anon")
        gt_root = os.path.join(acdc_root, "gt")

        def _sem_seg_path_from_image(image_path):
            # rgb_anon/fog/train/scene/name_rgb_anon.png -> gt/fog/train/scene/name_gt_labelTrainIds.png
            rel = os.path.relpath(image_path, image_root_base)
            rel = rel.replace("_rgb_anon.png", "_gt_labelTrainIds.png")
            return os.path.join(gt_root, rel)

        for w in weathers:
            # ---- SemSeg-only variant.
            gt_dir = os.path.join(gt_root, w, "train")
            img_dir = os.path.join(image_root_base, w, "train")
            semseg_name = f"acdc_{w}_semseg"

            def _load_semseg(img_dir=img_dir, gt_dir=gt_dir):
                out = []
                for scene in sorted(os.listdir(gt_dir)):
                    scene_dir = os.path.join(gt_dir, scene)
                    if not os.path.isdir(scene_dir):
                        continue
                    for fn in sorted(os.listdir(scene_dir)):
                        if not fn.endswith("_gt_labelTrainIds.png"):
                            continue
                        stem = fn.replace("_gt_labelTrainIds.png", "")
                        img_path = os.path.join(img_dir, scene, f"{stem}_rgb_anon.png")
                        if not os.path.isfile(img_path):
                            continue
                        out.append({
                            "file_name": img_path,
                            "sem_seg_file_name": os.path.join(scene_dir, fn),
                        })
                return out

            DatasetCatalog.register(semseg_name, _load_semseg)
            MetadataCatalog.get(semseg_name).set(
                evaluator_type="sem_seg",
                ignore_label=255,
                image_root=img_dir,
                sem_seg_root=gt_dir,
                **cityscapes_meta,
            )

            # ---- MTL variant: det annotations + sem_seg_file_name.
            det_json = os.path.join(
                acdc_root, "gt_detection", w,
                f"instancesonly_{w}_train_gt_detection.json",
            )
            if not os.path.isfile(det_json):
                continue
            mtl_name = f"acdc_{w}_mtl"

            def _load_mtl(det_json=det_json, mtl_name=mtl_name):
                # Pass mtl_name so load_ACDC_json sets
                # thing_dataset_id_to_contiguous_id on the MTL metadata (needed
                # by COCOEvaluator to map contiguous model IDs back to JSON IDs).
                dicts = load_ACDC_json(det_json, image_root_base, dataset_name=mtl_name)
                out = []
                for d in dicts:
                    sp = _sem_seg_path_from_image(d["file_name"])
                    if os.path.isfile(sp):
                        d["sem_seg_file_name"] = sp
                        out.append(d)
                return out

            DatasetCatalog.register(mtl_name, _load_mtl)
            # Precompute the id mapping from cityscapes labels so COCOEvaluator
            # can rely on metadata even before _load_mtl is called.
            from .ACDC import dataset_id_to_contiguous_id as _acdc_id_map
            MetadataCatalog.get(mtl_name).set(
                thing_classes=thing_classes,
                thing_dataset_id_to_contiguous_id=dict(_acdc_id_map),
                json_file=det_json,
                image_root=image_root_base,
                sem_seg_root=gt_root,
                evaluator_type="coco_sem_seg",
                ignore_label=255,
                **cityscapes_meta,
            )

    try:
        _register_acdc_semseg_and_mtl(_root)
    except Exception as e:
        print(f"[builtin] acdc_*_semseg/_mtl not registered: {e}")