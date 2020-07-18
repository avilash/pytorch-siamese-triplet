import cv2
import matplotlib.pyplot as plt


def visualise(imgs, txts, dst):
    f, axs = plt.subplots(1, len(imgs), figsize=(24, 9))
    f.tight_layout()
    for ax, img, txt in zip(axs, imgs, txts):
        ax.imshow(img)
        ax.set_title(txt, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.95, bottom=0.)
    if dst is not "":
        plt.savefig(dst)
    plt.show()


def vis_with_paths(img_paths, txts, dst):
    imgs = []
    for img_path in img_paths:
        imgs.append(cv2.imread(img_path))
    visualise(imgs, txts, dst)


def vis_with_paths_and_bboxes(img_details, txts, dst):
    imgs = []
    for img_path, bbox in img_details:
        img = cv2.imread(img_path)
        if bbox is not None:
            img = img[bbox['top']:bbox['top'] + bbox['height'], bbox['left']:bbox['left'] + bbox['width']]
        imgs.append(img)
    visualise(imgs, txts, dst)
