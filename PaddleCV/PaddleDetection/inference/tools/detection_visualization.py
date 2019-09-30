import detection_result_pb2
import cv2
import sys

def get_color_map(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    color_map = [(palette[i * 3 + 0], palette[i * 3 + 1], palette[i * 3 + 2]) for i in range(n)]
    return color_map

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python detection_visualization.py image image_result.pb")
    else:		
        text_color = get_color_map(100)
        text_thickness = 2 
        text_scale = 0.5 
        with open(sys.argv[2], "rb") as f:
            detection_result = detection_result_pb2.DetectionResult()
            detection_result.ParseFromString(f.read())
            img = cv2.imread(sys.argv[1])
            for box in detection_result.detection_boxes:
                box_class = getattr(box, 'class')
                text_class = "Class: %d" % (box_class)
                text_score = "Score: %.4f" % (box.score)
                text_point1 = (int(box.left_top_x), int(box.left_top_y - 30))
                text_point2 = (text_point1[0], text_point1[1] + 15)

                ptLeftTop = (int(box.left_top_x), int(box.left_top_y))
                ptRightBottom = (int(box.right_bottom_x), int(box.right_bottom_y))
                box_thickness = 2
                cv2.rectangle(img, ptLeftTop, ptRightBottom, text_color[box_class], box_thickness, 8)
                if text_point1[1] < 0:
                    text_point1 = (int(box.left_top_x), int(box.right_bottom_y + 15))
                    text_point2 = (text_point1[0], text_point1[1] + 15)
                cv2.putText(img, text_class, text_point1, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color[box_class], text_thickness)
                cv2.putText(img, text_score, text_point2, cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color[box_class], text_thickness)
            cv2.imwrite(sys.argv[1] + ".png", img)
