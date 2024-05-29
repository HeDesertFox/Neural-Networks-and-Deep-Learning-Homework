import numpy as np

from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose


def load_data(model, image):
    cfg = model.cfg.copy()
    test_pipeline = get_test_pipeline_cfg(cfg)
    if isinstance(image, np.ndarray):
        test_pipeline[0].type = "mmdet.LoadImageFromNDArray"
    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(dict(img_path=image))
    data["inputs"] = [data["inputs"]]
    data["data_samples"] = [data["data_samples"]]
    return data
