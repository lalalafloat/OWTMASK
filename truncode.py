# 改 syncbn 成 BN， 否则会RuntimeError: Default process group has not been initialized, please make sure to call init_process_group.

from detectron2.config import LazyConfig

def convert_py2yaml():
    config_file = 'configs/new_baselines/mask_rcnn_R_101_FPN_400ep_LSJ_OWT.py'
    cfg = LazyConfig.load(config_file)
    LazyConfig.save(cfg, "projects/OWT-Mask/configs/mask_rcnn_R_101_FPN_400ep_LSJ_OWT.yaml")
    # 转不了


if __name__ == '__main__':
    convert_py2yaml()
    pass