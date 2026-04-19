import numpy as np
import cv2

class MotionDetection_module:
    def __init__(self, config):
        self.md_h = config["md_h"]
        self.md_v = config["md_v"]
        self.bufnum = config["bufnum"]
        self.update_period = config["update_period"]
        self.pix_thresh = config["pix_thresh"]
        self.num_thresh = config["num_thresh"]
        self.reset()

    def update_buffer(self, img):
        if self.buf_isfull != 1:
            proc_img = self.preprocess(img)
            self.buffer[self.buf_idx] = proc_img
            self.buf_idx += 1
            if self.buf_idx == self.bufnum:
                self.buf_isfull = 1
                self.buf_idx = 0
        elif self.buf_cnt > self.update_period:
            proc_img = self.preprocess(img)
            self.buffer[self.buf_idx] = proc_img
            self.buf_idx = (self.buf_idx + 1) % self.bufnum
            self.buf_cnt = 0
        else:
            self.buf_cnt += 1

    def preprocess(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bin_img = cv2.resize(gray_img, (self.md_h, self.md_v), interpolation=cv2.INTER_AREA)
        return bin_img

    def detect(self, img):
        proc_img = self.preprocess(img)
        ave_buffer = np.mean(self.buffer, axis=0)
        diff = cv2.absdiff(proc_img, ave_buffer.astype(np.uint8))
        diff_norm = diff.astype(np.float32) / 255
        mask = (diff_norm > self.pix_thresh).astype(np.uint8)
        if np.sum(mask) > self.num_thresh:
            return 1
        else:
            return 0

    def reset(self):
        self.buf_cnt = 0
        self.buf_idx = 0
        self.buf_isfull = 0
        self.buffer = np.zeros((self.bufnum, self.md_v, self.md_h), dtype=np.uint8)