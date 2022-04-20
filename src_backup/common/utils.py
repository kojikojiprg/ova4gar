import matplotlib.pyplot as plt


def split_rgb(frame):
    r = frame.copy()
    g = frame.copy()
    b = frame.copy()

    r[:, :, (1, 2)] = 0
    g[:, :, (0, 2)] = 0
    b[:, :, (0, 1)] = 0
    return r, g, b


def show_img(img, cmap=None, is_save=False, save_name=None):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')

    if is_save:
        plt.savefig(save_name)

    plt.show()
