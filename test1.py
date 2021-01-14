import numpy as np
import cv2
import pytesseract
from PIL import Image

#https://codelearn.io/sharing/ai-phat-hien-nguoi-xam-nhap-p1

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'

def _getCannyImage(img):
    # chuyển hình sang màu gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # làm ảnh mờ
    # blur = cv2.GaussianBlur(gray, (7,7), 1)
    blur = cv2.bilateralFilter(gray, 20, 90, 90)

    #lấy viền của vật thể trong ảnh
    edges = cv2.Canny(blur, 55, 150)
    return edges


def _smoothyImage(img):
    blur = cv2.bilateralFilter(img, 50, 200, 75)
    return blur

def _getImageFromCam():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = "opencv_carPlate_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

# _getImageFromCam()

path = "./images/car_3.jpg"
# path = "opencv_carPlate_0.png"
# Đọc hình original
image = cv2.imread(path)
# rSize = cv2.resize(image, (1024, 770))

edges = _getCannyImage(image)

if edges.shape[1] > 768:
    edgesResize = cv2.resize(edges, (round(edges.shape[1]/2), round(edges.shape[0]/2)))
else:
    edgesResize = cv2.resize(edges, (edges.shape[1], edges.shape[0]))
cv2.imshow("canny", edgesResize)
cnts, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
# print(type(cnts))

image_copy = cv2.drawContours(image_copy, cnts, -1, (255, 0, 255), 2)
# cv2.imshow('img copy 1', image_copy)
cnts_sort = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
# print(cnts_sort)

image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, cnts_sort, -1, (255, 0, 255), 2)
# image_copy = cv2.resize(image_copy, (350, 450))
# cv2.imshow('img copy 2', image_copy)


def _drawBoundingBox(img):
    # x, y, w, h = cv2.boundingRect(cnt)
    image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


def _drawcontour(img, edges_count):
    rect = cv2.minAreaRect(edges_count)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    return img


def _cropPlateImage(img, edges_count):

    rect = cv2.minAreaRect(edges_count)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # # cv2.line(img, (box[1][0], box[1][1]), (box[2][0], box[2][1]), (0, 255, 255), 2)
    # # pts1 = np.array([box[2], box[3], box[0], box[1]], dtype="float32")
    # pts1 = np.array([box], dtype="float32")
    # pts2 = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # plate = cv2.warpPerspective(img, matrix, (w, h))
    print(rect)
    if abs(rect[2]) > 80:
        plate = img[y:y+h, x:x+w]
    else:
        plate = img[y:y+h, x:x+w]
        h1, w1 = plate.shape[0], plate.shape[1]
        M5 = cv2.getRotationMatrix2D(center=(w1 / 2, h1 / 2), angle=(rect[2]), scale=1)
        plate = cv2.warpAffine(plate, M5, (w1, h1))
    return plate



imgOrigin = image.copy()
imgcp = image.copy()
# Vẽ bounding box cho contours và cắt hình biển số ra
plate = None
for c in cnts_sort:
    perimeter = cv2.arcLength(c, True)
    edges_count = cv2.approxPolyDP(c, 0.02 * perimeter, True)
    if len(edges_count) == 4:
        # print(c[0][0])
        x, y, w, h = cv2.boundingRect(c)
        imgOrigin = _drawcontour(imgOrigin, edges_count)
        imgOrigin = _drawBoundingBox(imgOrigin)
        plate = _cropPlateImage(imgcp, edges_count)
        break

# print(imgOrigin.shape)
h, w = imgOrigin.shape[0], imgOrigin.shape[1]
if w < 768:
    imgOrigin = cv2.resize(imgOrigin, (w, h))
else:
    imgOrigin = cv2.resize(imgOrigin, (round(w/3), round(h/3)))
cv2.imshow('imgOrigins', imgOrigin)
cv2.imshow('plate', plate)

# plate = cv2.resize(plate,(255, 150))

imgGray = cv2.cvtColor(plate, cv2.COLOR_BGR2RGB)
# print(imgGray)
# cv2.imshow('plate gray', imgGray)
cv2.imwrite("plate.png", imgGray)



text = pytesseract.image_to_string(Image.open("plate.png"))
print(pytesseract.image_to_boxes(Image.open("plate.png")))
print("Detected Plate Character is:\n", text)




cv2.waitKey(0)