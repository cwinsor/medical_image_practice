
import os
import argparse
from glob import glob

import numpy as np
import pandas as pd

# visualization
import cv2


# PyTorch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', default='bar')

    args = parser.parse_args()

    return args


# zona 16
def read_image(name):
    img = cv2.imread(name, cv2.IMREAD_ANYDEPTH) / 65535.0
    img = 255 * ((img - img.min()) / (img.max() - img.min()))
    img = img.astype(np.uint8)
    return img


# zona 21
def df_preprocessing(df, globbed_file_list):
    """The preprocessing steps applied to get column information"""

    # 1. Get Case-ID as a column (str and int)
    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])
    df["case_id"] = df["id"].apply(lambda x: int(x.split("_", 2)[0].replace("case", "")))

    # 2. Get Day as a column
    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])
    df["day_num"] = df["id"].apply(lambda x: int(x.split("_", 2)[1].replace("day", "")))

    # 3. Get Slice Identifier as a column
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    # 4. Get full file paths for the representative scans
    df["_partial_ident"] = (
        globbed_file_list[0].rsplit("/", 4)[0]
        + "/"
        + df["case_id_str"]  # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
        + "/"
        + df["case_id_str"]  # .../case###/
        + "_"
        + df["day_num_str"]
        + "/scans/"  # .../case###_day##/
        + df["slice_id"]
    )  # .../slice_####
    _tmp_merge_df = pd.DataFrame(
        {
            "_partial_ident": [x.rsplit("_", 4)[0] for x in globbed_file_list],
            "f_path": globbed_file_list,
        }
    )
    df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[1]))
    df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_", 4)[2]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[3]))
    df["px_spacing_w"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_", 4)[4]))

    # 7. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
    new_col_order = [
        "id",
        "f_path",
        "slice_h",
        "slice_w",
        "px_spacing_h",
        "px_spacing_w",
        "case_id_str",
        "case_id",
        "day_num_str",
        "day_num",
        "slice_id",
    ]
    new_col_order = [_c for _c in new_col_order if _c in df.columns]
    df = df[new_col_order]
    return df


# zona 81
def get_nearby_slices(id_, case_id_str, day_num_str, case_length, num_slices=3, num_strides=1):
    slice_idx = int(id_.split("_")[-1])
    get_idxs = np.arange(slice_idx - num_slices//2 * num_strides,
                         slice_idx + num_slices//2 * num_strides + 1, num_strides)  # -7 -5 -3 -1 1 3 5 7 9

    min_idx = 2 if slice_idx % 2 == 0 else 1
    if case_length % 2 == 0:
        max_idx = case_length if slice_idx % 2 == 0 else case_length - 1
    else:
        max_idx = case_length if slice_idx % 2 != 0 else case_length - 1

    get_idxs = np.clip(get_idxs, min_idx, max_idx)
    get_ids = [f"{case_id_str}_{day_num_str}_slice_{slice_idx:04d}" for slice_idx in get_idxs]
    return get_ids


class CFG:
    ARCH = ['DeepLabV3Plus',
            'DeepLabV3PlusFix',
            # 'DeepLabV3PlusFix',
            'Unet',
            'UnetPlusPlusFix',
            'UnetPlusPlus',
            'Unet',
            'UnetPlusPlus'
            ]
    ENCODER = ['convnext_xlarge_in22ft1k',
               'tf_efficientnetv2_l',
               # 'ecaresnet269d',
               'convnext_base',
               'convnext_xlarge_in22ft1k',
               'tf_efficientnetv2_l',
               'convnext_base',
               'ecaresnet269d'
               ]
    WEIGHTS = [
       '../input/newgiseg/deeplabv3plus_convnext_xlarge_17_608_fold-1_e28.pth',
       '../input/uwmgiweights/deeplabv3plus_v2l_17_608_fold-1_e28.pth',
       # '../input/uwmgiweights/deeplabv3plus_ecaresnet269d_17_608_long_fold-1_e98.pth',
       '../input/uwmgiseg/weights/convnext_base_acs_unet_epoch30.pth',
       '../input/uwmgiweights/unetplus_convnext_xlarge_fold-1_e28.pth',
       '../input/uwmgiweights/unetplus_v2l_fold-1_e28.pth',
       '../input/uwmgiseg/weights/convnext_base_unet_epoch10.pth',
       '../input/uwmgiweights/unetplus_ecaresnet269d_fold-1_e98.pth'
              ]
    NUM_CLASSES = 3

    img_size = [[640, 640], [640, 640],
                # [640, 640],
                [512, 512], [640, 640], [640, 640], [512, 512],
                [640, 640]]
    slices = [17, 17,
              # 17,
              33, 9, 9, 5, 9]
    strides = [2, 2,
               # 3,
               1, 2, 3, 1, 2]
#     THRESHOLDS = [0.3, 0.3, 0.4]
    THRESHOLDS = [0.3, 0.3, 0.375]


class GISegDataset(Dataset):
    def __init__(self, df, target_size):
        super().__init__()
        resized_h, resized_w = target_size
        df["case_day_str"] = df["case_id_str"] + "_" + df["day_num_str"]
        self.cases_length = df["case_day_str"].value_counts().to_dict()
        self.images_dict = {id: f_path for id, f_path in zip(df["id"].values, df["f_path"].values)}
        self.aug512 = A.Compose([A.Resize(512, 512), ToTensorV2()])
        self.aug608 = A.Compose([A.Resize(608, 608), ToTensorV2()])
        self.aug640 = A.Compose([A.Resize(640, 640), ToTensorV2()])
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        center_id = info["id"]
        case_length = self.cases_length.get(info["case_day_str"])
        case_id = info["case_id_str"]
        day = info["day_num_str"]

        ids172 = get_nearby_slices(center_id, case_id, day, case_length,
                                   num_slices=17, num_strides=2)
#         ids173 = get_nearby_slices(center_id, case_id, day, case_length,
#                                    num_slices=17, num_strides=3)
        ids331 = get_nearby_slices(center_id, case_id, day, case_length,
                                   num_slices=33, num_strides=1)
        ids92 = get_nearby_slices(center_id, case_id, day, case_length,
                                  num_slices=9, num_strides=2)
        ids93 = get_nearby_slices(center_id, case_id, day, case_length,
                                  num_slices=9, num_strides=3)
#         img_ids = set(ids172 + ids173 + ids331 + ids92 + ids93)
        img_ids = set(ids172 + ids331 + ids92 + ids93)

        imgs_dict = {id_: read_image(self.images_dict.get(id_)) for id_ in img_ids}
        h, w = imgs_dict.get(ids172[0]).shape

        img172 = np.stack([imgs_dict.get(id_) for id_ in ids172], axis=-1)
        img172 = self.aug640(image=img172)["image"].float() / 255.

#         img173 = np.stack([imgs_dict.get(id_) for id_ in ids173], axis=-1)
#         img173 = self.aug640(image=img173)["image"].float() / 255.

        img331 = np.stack([imgs_dict.get(id_) for id_ in ids331], axis=-1)
        img331 = self.aug512(image=img331)["image"].float() / 255.
        img51 = img331[14:19]

        idx92 = []
        # idx93 = []
        for i in ids92:
            idx92.append(ids172.index(i))
        img92 = img172[idx92]
#         for i in ids93:
#             idx93.append(ids173.index(i))
#         img93 = img173[idx93]

        img93 = np.stack([imgs_dict.get(id_) for id_ in ids93], axis=-1)
        img93 = self.aug640(image=img93)["image"].float() / 255.

#         return img172, img173, img331, img92, img93, img51, center_id, h, w
        return img172, img331, img92, img93, img51, center_id, h, w


def main():
    args = parse_args()

    # DATA_DIR = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"
    DATA_DIR = "/mnt/d/code_medimg_kag_uw_madison/medical_image_uw_madison/project/Kaggle-UWMGIT/data/tract/"
    TEST_DIR = os.path.join(DATA_DIR, "test")
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

    sub_df = pd.read_csv(SUB_CSV)

    if not len(sub_df):
        # Infer on train cases
        # debug = True
        sub_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        sub_df = sub_df.drop(columns=['class', 'segmentation']).drop_duplicates()
        paths = glob(TRAIN_DIR + '/**/*png', recursive=True)
        sub_df = df_preprocessing(sub_df, paths)
        cases = sub_df["case_id_str"].unique()[:1]
        sub_df = sub_df[sub_df["case_id_str"].isin(cases)].reset_index(drop=True)
    else:
        assert False, "ZONA - does TEST_DIR exist?"
        # debug = False
        sub_df = sub_df.drop(columns=['class', 'predicted']).drop_duplicates()
        paths = glob(TEST_DIR + '/**/*png', recursive=True)
        sub_df = df_preprocessing(sub_df, paths)
        sub_df = sub_df.reset_index(drop=True)

    test_dataset = GISegDataset(sub_df, CFG.img_size[0])
    test_loader = DataLoader(
        test_dataset, batch_size=4,
        num_workers=2, shuffle=False, pin_memory=True)
    # pred_strings, pred_ids, pred_classes = inference(
    #     all_models, test_loader)

    # ###################################################
    # Now that we have DataLoader and Dataset...
    # for training we follow the "cars" example from Segmentation Models PyTorch site
    print("here")
    


if __name__ == '__main__':
    main()
