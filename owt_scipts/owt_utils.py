import os

import numpy as np

from typing import List

from pycocotools.mask import encode


def store_TAOnpz(predictions, input_img_path: str, valid_classes: List[int], npz_outdir: str):
    """
    Store all the proposals in one frame.
    Output in `.npz` format:

    - List of Dict:
        {"category_id": (int),
         "bbox": [x1, y1, x2, y2],
         "instance_mask": {"size": [img_h, img_w], "counts": rle_str}
         "score": (float),
         "bg_score": (float),
         "objectness": (float),
         "embeddings": (np.array), shape=(2048,)
    """
    frame_name = input_img_path.split('/')[-1].replace('.jpg', '.npz').replace('.png', '.npz')
    frame_name = frame_name.split('-')[-1]  # specifically for bdd-100k data
    npz_outpath = os.path.join(npz_outdir, frame_name)
    if os.path.exists(npz_outpath):  # [ATTN] 之前生成的就不生成了
        return
    output = list()

    pred_classes = predictions['instances'].pred_classes
    # print('pred_classes = {}'.format(pred_classes))  # 全是0
    for i in range(len(pred_classes)):
        proposal = dict()
        if pred_classes[i] in valid_classes:  # 如果改成cls_agn记得修改 valid_classes 为[0]  [TODO]
            proposal['category_id'] = pred_classes[i].cpu().numpy().tolist()
            # print('category_id = {}'.format(proposal['category_id']))
            bbox = predictions['instances'].pred_boxes[i].tensor.cpu().numpy().tolist()[0]
            proposal['bbox'] = [int(b) for b in bbox]  # Convert bbox coordinates to int

            proposal['score'] = predictions['instances'].scores[i].cpu().numpy().tolist()
            proposal['bg_score'] = predictions['instances'].bg_scores[i].cpu().numpy().tolist()
            proposal['objectness'] = predictions['instances'].objectness[i].cpu().numpy().tolist()
            embeddings = predictions['instances'].embeds[i].cpu().numpy()
            proposal['embeddings'] = (embeddings * 1e4).astype(np.uint16).tolist()
            output.append(proposal)
    # print('len_output = {}'.format(len(output)))
    np.savez_compressed(npz_outpath, output)
