# eval.py

def iou(boxA, boxB):
    """
    box format: [x, y, width, height]
    computes IoU between two bounding boxes
    """

    ax1, ay1, aw, ah = boxA
    bx1, by1, bw, bh = boxB

    ax2 = ax1 + aw
    ay2 = ay1 + ah
    bx2 = bx1 + bw
    by2 = by1 + bh

    # Intersection rectangle
    x_left = max(ax1, bx1)
    y_top = max(ay1, by1)
    x_right = min(ax2, bx2)
    y_bottom = min(ay2, by2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    areaA = aw * ah
    areaB = bw * bh

    iou_result = inter_area / float(areaA + areaB - inter_area)
    return round(iou_result, 4)
