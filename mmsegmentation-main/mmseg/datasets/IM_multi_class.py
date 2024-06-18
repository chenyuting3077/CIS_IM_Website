# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class IMMultiClass(BaseSegDataset):
    """ADE20K dataset.

    """
    METAINFO = dict(
        classes=('Non-IM', 'IM-1', 'IM-2', 'IM-3'),
        palette=[[0, 0, 0], [0, 0, 255], [255, 0, 0], [0, 255, 0]],
    )

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 seg_map_suffix='_gtFine_labelIds.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)