# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path as osp

from core.config import cfg

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(cfg.ROOT_DIR, '..', '..', 'data', 'raw'))

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'clevr_mini': {
        IM_DIR:
            osp.join(_DATA_DIR, 'CLEVR_mini/images'),
        ANN_FN:
            osp.join(_DATA_DIR, 'CLEVR_mini/CLEVR_mini_coco_anns.json'),
    },
    'clevr_original_train': { # both clevr_original_train and clevr_original_val are used for testing mask_rcnn
        IM_DIR:
            osp.join(_DATA_DIR, 'CLEVR_v1.0/images/train'),
        ANN_FN:
            osp.join(_DATA_DIR, 'CLEVR_v1.0/scenes/CLEVR_train_scenes.json'),
    },
    'clevr_original_val': {
        IM_DIR:
            osp.join(_DATA_DIR, 'CLEVR_v1.0/images/val'),
        ANN_FN:
            osp.join(_DATA_DIR, 'CLEVR_v1.0/scenes/CLEVR_val_scenes.json'),
    },
}