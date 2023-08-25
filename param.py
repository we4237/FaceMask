from typing import Tuple

import cv2
import numpy as np


def _post_process_face_neck_mask(
            self, prediction: np.ndarray, inversed_matrix: np.ndarray,
            frame_shape: Tuple[int, int],
            face_shape: Tuple[int, int],
            post_param: Tuple[int, int, int, int],
            circle_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对模型结果进行后处理

        Args:
            prediction (np.ndarray): 模型输出结果
            inversed_matrix (np.ndarray): 仿射变换矩阵
            frame_shape (Tuple[int, int]): 原始的驱动帧的高宽(height * width)
            face_shape (Tuple[int, int]): 人脸检测的人脸的高宽(heigth * width)
            post_param (Tuple[int, int, int]): 人脸分割形态学后处理参数
            circle_mask (np.ndarray): 人脸有效圆形mask,处理脖子处贴合部分
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                返回face_mask 和 neck_sff_mask
                face_mask为(N, origin_frame_height, origin_frame_width)
                neck_sff_mask为(N, face_height, face_width, 2)
        """
        face_mask = self._get_face_mask(prediction=prediction)
        face_neck_mask = self._get_face_neck_mask(prediction=prediction) 
        neck_mask = self._get_neck_mask(prediction=prediction)
        sff_mask = face_neck_mask.copy()

        face_neck_mask = (face_neck_mask * circle_mask).astype(np.uint8)
        img_white = np.full((face_shape[0], face_shape[1]), 1.0, dtype=float)

        # neck_sff_mask = neck_mask.copy()
        face_neck_mask_up, face_neck_mask_down, face_mask_down = face_neck_mask[:face_shape[
            0] // 3, :], face_neck_mask[face_shape[0] //
                                        3:, :], face_mask[face_shape[0] //
                                                          3:, :]

        kernel = np.ones((int(face_shape[0] * 0.02), int(face_shape[0] * 0.02)),
                         np.uint8)
        face_neck_mask_up_pad = np.concatenate((np.zeros_like(face_neck_mask_up), face_neck_mask_up), axis = 0)
        face_neck_mask_up = self._get_morph_func(post_param[0], kernel)(face_neck_mask_up_pad)[face_neck_mask_up.shape[0]:, ]
        face_neck_mask_down_face = self._get_morph_func(post_param[1], kernel)(face_neck_mask_down)
        face_neck_mask_down_neck = self._get_morph_func(post_param[2], kernel)(face_neck_mask_down)
        face_mask_down = self._get_morph_func(post_param[1], kernel)(face_mask_down)
        
        mask_up = face_neck_mask_up
        mask_down = face_neck_mask_down_face * (
            face_mask_down /
            255) + face_neck_mask_down_neck * (1 - face_mask_down / 255)
        mask = np.concatenate([mask_up, mask_down], axis=0)

        mask[int(-0.0625 * face_shape[0]):,] = 0
        img_white *= mask
        neck_sff_mask = np.zeros((face_shape[0], face_shape[0]), dtype=np.uint8)
        left, right, up, down, neck_down = 0.125, 0.875, 0.50 - (
            0.03125 + 0.09375), 0.96875 - (0.125 + 0.09375), 0.96875
        neck_sff_mask[int(face_shape[0] * down):int(face_shape[0] * neck_down),
                      int(face_shape[0] *
                          left):int(face_shape[0] * right)] = neck_mask[
                              int(face_shape[0] * down):int(face_shape[0] *
                                                            neck_down),
                              int(face_shape[0] * left):int(face_shape[0] *
                                                            right)]
        neck_sff_mask[int(face_shape[0] * up):int(face_shape[0] * down),
                      int(face_shape[0] * left):int(face_shape[0] *
                                                    right)] = 255

        neck_sff_mask = cv2.GaussianBlur(
            neck_sff_mask, ((int(face_shape[0] * 0.08 * 2) // 2 * 2 + 1),
                            (int(face_shape[0] * 0.08 * 2) // 2 * 2 + 1)),
            cv2.BORDER_DEFAULT)
        sff_mask = cv2.dilate(
            sff_mask,
            np.ones((int(face_shape[0] * 0.04), int(face_shape[0] * 0.04)),
                    np.uint8),
            iterations=1)
        neck_sff_mask = np.stack((neck_sff_mask, sff_mask), axis=-1)
        face_mask2 = cv2.warpAffine(img_white, inversed_matrix,
                                    (frame_shape[1], frame_shape[0]))
        k = int((np.sum(face_mask2) / 255)**0.5 * 0.10)
        if k % 2 == 0:
            k += 1
        #face_mask = (cv2.GaussianBlur(face_mask2,(k,k),0) * 255).astype(np.uint8)[box[0]:box[1], box[2]:box[3]]
        face_mask = (cv2.GaussianBlur(face_mask2, (k, k), 0)).astype(np.uint8)
        # face_mask = face_mask2
        #cv2.imwrite('temp/face_mask.png', (face_mask * 255).astype(np.uint8))
        #cv2.imwrite('temp/face_mask.png', face_mask)
        # cv2.imwrite('temp/erosion_mask.png', erosion_mask)
        return (face_mask, neck_sff_mask)





def _get_fixed_circle_mask(self, face_shape: Tuple[int, int], post_loc: int) -> np.ndarray:
    """生成圆形mask,圆形内的mask有效
    Args:
        face_shape (Tuple[int, int]): 人脸大小
        post_loc (int): 标记使用圆形mask的范围
    Returns:
        np.ndarray: circle_mask
    """
    y, x = np.ogrid[0:face_shape[1], 0:face_shape[0]]
    center_x, center_y, radius = \
        int(face_shape[0] * 0.5), int(face_shape[1] * (0.639 + post_loc / 64.0)), int(face_shape[0] * 0.391)
    circle_mask = ((x-center_x)**2+(y-center_y)**2<=radius**2)
    circle_mask = (circle_mask*255).astype(np.uint8)
    circle_mask[:-int(face_shape[0] * (1 / 16.0 - post_loc / 64.0)),:,] = 255
    circle_mask = ((cv2.GaussianBlur(circle_mask,(35,35),0)) / 255.0).astype(np.float32)
    return circle_mask