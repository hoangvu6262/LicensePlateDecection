# # def nms(bounding_boxes, confidence_score, threshold):
# #     # If no bounding boxes, return empty list
# #     if len(bounding_boxes) == 0:
# #         return [], []
# #
# #     # Bounding boxes
# #     boxes = np.array(bounding_boxes)
# #
# #     # coordinates of bounding boxes
# #     start_x = boxes[:, 0]
# #     start_y = boxes[:, 1]
# #     end_x = boxes[:, 2]
# #     end_y = boxes[:, 3]
# #
# #     # Confidence scores of bounding boxes
# #     score = np.array(confidence_score)
# #
# #     # Picked bounding boxes
# #     picked_boxes = []
# #     picked_score = []
# #
# #     # Compute areas of bounding boxes
# #     areas = (end_x - start_x + 1) * (end_y - start_y + 1)
# #
# #     # Sort by confidence score of bounding boxes
# #     order = np.argsort(score)
# #
# #     # Iterate bounding boxes
# #     while order.size > 0:
# #         # The index of largest confidence score
# #         index = order[-1]
# #
# #         # Pick the bounding box with largest confidence score
# #         picked_boxes.append(bounding_boxes[index])
# #         picked_score.append(confidence_score[index])
# #
# #         # Compute ordinates of intersection-over-union(IOU)
# #         x1 = np.maximum(start_x[index], start_x[order[:-1]])
# #         x2 = np.minimum(end_x[index], end_x[order[:-1]])
# #         y1 = np.maximum(start_y[index], start_y[order[:-1]])
# #         y2 = np.minimum(end_y[index], end_y[order[:-1]])
# #
# #         # Compute areas of intersection-over-union
# #         w = np.maximum(0.0, x2 - x1 + 1)
# #         h = np.maximum(0.0, y2 - y1 + 1)
# #         intersection = w * h
# #
# #         # Compute the ratio between intersection and union
# #         ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
# #
# #         left = np.where(ratio < threshold)
# #         order = order[left]
# #
# #     return picked_boxes, picked_score
#
# #---------------------------Làm việc với plateImage vừa lấy ra được-----------------------------------
# imgOrigin = plate.copy()
# imgOriginResize = cv2.resize(imgOrigin, (300, 100))
# smoothy = _smoothyImage(imgOriginResize)
# cannyPlate = _getCannyImage(smoothy)
#
# #lấy ra list các contour có diện tích lớn nhất từ plateImage vừa cắt ra
# contours, hierarchy = cv2.findContours(cannyPlate.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# image_copy = smoothy.copy()
#
# #sắp xếp theo diện tích
# image_copy = cv2.drawContours(image_copy, cnts, -1, (255, 0, 255), 2)
# contours_sort = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
#
#
# image_copy = smoothy.copy()
# image_copy = cv2.drawContours(image_copy, contours_sort, -1, (255, 0, 255), 2)
#
# number = None
# for c in contours_sort:
#     x, y, w, h = cv2.boundingRect(c)
#     img = _drawcontour(smoothy, c)
#     # img = _drawBoundingBox(smoothy)
#     number = _cropPateImage(img)
#
#
# cv2.imshow('img', img)
# # cv2.imshow('number', number)
#
#
#
# # Draw parameters
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# thickness = 2
#
#
# # Vẽ bounding box cho contours
# boundingBoxes = []
# for i in contours_sort[:10]:
#     x, y, w, h = cv2.boundingRect(i)
#     x1, y1, x2, y2 = x, y, x+w, y+h
#     boundingBoxes.append((x1, y1, x2, y2))
#
# confidence_score = [0.5, 0.9, 0.7]
#
#
# org = smoothy.copy()
# # vẽ bounding boxes và confidence score
# for (start_x, start_y, end_x, end_y), confidence in zip(boundingBoxes, confidence_score):
#     (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
#     cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
#     # cv2.rectangle(org, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
#     # cv2.putText(org, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
#
#
# # picked_boxes, picked_score = nms(boundingBoxes, confidence_score, 0.4)
#
#
# # for (start_x, start_y, end_x, end_y), confidence in zip(picked_boxes, picked_score):
# #     (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
# #     cv2.rectangle(imgOrigin, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
# #     # cv2.rectangle(imgOrigin, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2)
# #     # cv2.putText(smoothy, str(confidence), (start_x, start_y), font, font_scale, (0, 0, 0), thickness)
#
# # Show image
#
# # cv2.imshow('org', org)
# # cv2.imshow('NMS', smoothy)


import cv2

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_test_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
