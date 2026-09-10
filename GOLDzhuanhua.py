import base64
import json
import os
import os.path as osp

import numpy as np
from PIL import Image
from labelme import utils
from labelme.logger import logger

def main():
    json_dir = r"./data/images"     # JSON 标注文件路径
    out_dir = r"./data/"             # 输出路径
    img_out_dir = osp.join(out_dir, "image")
    mask_out_dir = osp.join(out_dir, "mask")

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # 固定类别标签
    label_name_to_value = {
        "_background_": 0,
        "EPB": 1,
        "EIA": 2
    }

    for file_name in os.listdir(json_dir):
        if file_name.endswith(".json"):
            json_path = osp.join(json_dir, file_name)

            with open(json_path, "r") as f:
                data = json.load(f)

            image_data = data.get("imageData")
            if not image_data:
                image_path = osp.join(json_dir, data["imagePath"])
                with open(image_path, "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")

            img = utils.img_b64_to_arr(image_data)

            image_save_path = osp.join(img_out_dir, f"{file_name.split('.')[0]}_img.png")
            img_pil = Image.fromarray(img)
            if img_pil.mode == 'RGBA':
                img_pil = img_pil.convert('RGB')
            img_pil.save(image_save_path)

            for shape in data["shapes"]:
                label_name = shape["label"]
                if label_name not in label_name_to_value:
                    raise ValueError(f"标签 '{label_name}' 不在 label_name_to_value 中定义。")

            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], label_name_to_value=label_name_to_value
            )

            # ❗保存为单通道 0/1/2 的灰度图，而非伪彩色图
            mask_save_path = osp.join(mask_out_dir, f"{file_name.split('.')[0]}_mask.png")
            Image.fromarray(lbl.astype(np.uint8)).save(mask_save_path)

            logger.info(f"Processed {file_name}")

    print("✅ 标签映射:", label_name_to_value)

if __name__ == "__main__":
    main()
