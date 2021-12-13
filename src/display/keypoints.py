import cv2


def draw_keypoints(frame, keypoints, scores):
    l_pair = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12),  # Body
        (11, 13), (12, 14), (13, 15), (14, 16)
    ]
    p_color = [
        # Nose, LEye, REye, LEar, REar
        (0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
        # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
        (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
        # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127), (0, 255, 255)
    ]
    line_color = [
        (0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
        (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
        (77, 222, 255), (255, 156, 127),
        (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)
    ]

    img = frame.copy()
    part_line = {}
    vis_thresh = 0.4
    # Draw keypoints
    for n in range(len(keypoints)):
        if scores[n] <= vis_thresh:
            continue
        cor_x, cor_y = int(keypoints[n, 0]), int(keypoints[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(img, (cor_x, cor_y), 3, p_color[n], -1)
    # Draw limbs
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(img, start_xy, end_xy, line_color[i], 2 * int(scores[start_p] + scores[end_p]) + 1)

    return img
