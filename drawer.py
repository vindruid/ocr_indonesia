import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class DrawOCR(object):

    def __init__(self, args={}):
        self.COLORS = args.get(
            "colors", [(0, 120, 0), (120, 0, 50), (0, 100, 100)])

    def show_list_images(self, list_img):
        list_img = [x for x in list_img if x is not None]
        num_of_img = len(list_img)
        f, ax = plt.subplots(figsize=(10, 7), ncols=1, nrows=num_of_img)

        if num_of_img > 1:
            for i, im in enumerate(list_img):
                ax[i].imshow(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
        elif num_of_img == 0:
            ax.imshow(cv2.cvtColor(list_img[0], cv2.COLOR_RGB2BGR))
        else:
            pass

        plt.show()

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 156) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    def plot_bboxes(self, image, boxes, labels):  # boxes and labels are dictionaries
        # Plot Box in Image
        img_show = image.copy()
        for idx, box in boxes.items():
            if box is not None:
                self.plot_one_box(
                    box, img_show, color=self.COLORS[idx], label=list(labels.values())[idx])

        # Show Images
        f, ax = plt.subplots(figsize=(10, 7))
        title = ""
        for lab in labels:
            if labels[lab] is not None:
                title += str(lab) + " : " + labels[lab] + ", "
        plt.title(title)
        ax.imshow(cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
        plt.show()

    def show_desc(self, image, boxes, labels, listdata, listlabel, tf=2, tl=3, show_plot=True):
        H, W = image.shape[:2]
        img_show = image.copy()

        for idx, box in boxes.items():
            if box is not None:
                self.plot_one_box(
                    box, img_show, color=self.COLORS[idx], label=list(labels.values())[idx])

        # Create White Description Layer
        desc = np.zeros((H, W, 3), np.uint8)
        desc.fill(255)
        for i, (dat, lab) in enumerate(list(zip(listdata, listlabel))):
            cv2.putText(desc, "{} : {}".format(lab, dat), (20, ((H // 8) * (i + 1))), 0, tl / 3.5,
                        [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)

        img_show = cv2.hconcat([img_show, desc])
        
        if show_plot:
            plt.figure(figsize=(20, 7))
            plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR))
            
        else:
            return img_show

