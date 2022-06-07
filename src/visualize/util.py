from numpy.typing import NDArray
from utility.video import concat_field_with_frame


def delete_time_bar(frame: NDArray, delete_hight: int = 20):
    return frame[delete_hight:]


def get_size(frame: NDArray, field: NDArray):
    cmb_img = concat_field_with_frame(frame, field)
    return cmb_img.shape[1::-1]
