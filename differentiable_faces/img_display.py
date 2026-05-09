import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision.utils import save_image
import glob
import re


def compute_rows_columns(num_images, image_per_row=6):
    if num_images <= image_per_row:
        rows = 1
        cols = num_images
    else:
        cols = image_per_row
        rows = int(num_images / cols) + 1
    return rows, cols


def display_images(path, nConds, num_per_row, target=False, image_size=3):

    if target == False:
        save_path = path + "all_current.png"
        img_list = [path + "current" + str(i) + ".png" for i in range(nConds)]
    else:
        save_path = path + "all_target.png"
        img_list = [path + "target" + str(i) + ".png" for i in range(nConds)]

    rows, cols = compute_rows_columns(len(img_list), num_per_row)
    width = cols * image_size
    height = rows * image_size

    axes = []
    fig = plt.figure(figsize=(width, height))
    for count, filename in enumerate(img_list):
        axes.append(fig.add_subplot(rows, cols, count + 1))
        img = mpimg.imread(filename)
        plt.grid("off")
        plt.axis("off")
        plt.imshow(img)
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


def display_faces(path, face_list, num_per_row=5, image_size=3):
    img_list = []
    for i in range(len(face_list)):
        face = face_list[i].image_torch
        save_image(face, path + str(i) + ".png")
        img_list.append(path + str(i) + ".png")

    rows, cols = compute_rows_columns(len(face_list), num_per_row)
    width = cols * image_size
    height = rows * image_size

    axes = []
    fig = plt.figure(figsize=(width, height))
    for count, filename in enumerate(img_list):
        axes.append(fig.add_subplot(rows, cols, count + 1))
        img = mpimg.imread(filename)
        plt.grid("off")
        plt.axis("off")
        plt.imshow(img)
    fig.tight_layout()
    plt.show()


def display_lf(path, image_size=3):
    img_list = sorted(glob.glob(path + "*.png"))
    rows, cols = compute_rows_columns(len(img_list), 6)
    width = cols * image_size
    height = rows * image_size

    fig, axes = plt.subplots(rows, cols, figsize=(width, height))
    for count, filename in enumerate(img_list):
        row = int(count / cols)
        col = int(count % cols)
        img = mpimg.imread(filename)
        title = re.search("loss_function/(.+?).png", filename).group(1)
        if rows == 1:
            axes[col].set_title("%s" % title)
            axes[col].axis("off")
            axes[col].imshow(img)
        else:
            axes[row, col].set_title("%s" % title)
            axes[row, col].axis("off")
            axes[row, col].imshow(img)

        plt.grid("off")
        plt.axis("off")

    fig.tight_layout()
    plt.show()
