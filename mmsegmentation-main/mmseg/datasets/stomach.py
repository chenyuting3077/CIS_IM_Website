# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class StomachDataset(BaseSegDataset):
    """ADE20K dataset.

    """
    METAINFO = dict(
        classes=('Background', 'IM_0', 'IM_1', "IM_2", "IM_3"),
        palette=[[0, 0, 0], [0, 0, 64], [0, 0, 128], [0, 0, 192],[0, 0, 255]]
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)