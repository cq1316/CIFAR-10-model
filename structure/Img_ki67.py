import cv2
import numpy as np
import torch


class Img_ki67():
    def __init__(self, infor_dict):
        self.infor = infor_dict
        src_img = np.array(infor_dict["src_img"])
        self.src_img = (((src_img - src_img.min()) / (src_img.max() - src_img.min())) * 255).astype(
            np.uint8)
        self.roi_img = self.src_img[self.infor["min_y"] - 3:self.infor["max_y"] + 4,
                       self.infor["min_x"] - 3:self.infor["max_x"] + 4]

    def get_class(self):
        return self.infor["class"]

    def get_id(self):
        return self.infor["id"]

    def make_resize(self, resize_length):
        # resize
        roi_resize_img = cv2.resize(self.roi_img, (resize_length, resize_length))
        # tensor
        self.transform_tensor = torch.FloatTensor([roi_resize_img])

    def get_tensor(self):
        return self.transform_tensor
