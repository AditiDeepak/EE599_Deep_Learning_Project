import cv2 as cv
import numpy as np

def color_filter(img, r, g, b):
    colors = [b, g, r]
    result = np.zeros(img.shape, dtype=np.uint8)
    for i in range(3):
        result[:, :, i] = np.where(img[:, :, i] < colors[i], 0, 255)
    return result.astype(np.uint8)

def test_colors(img):
    cv.imshow("test_colors", img)
    r = 100
    g = 100
    b = 100
    while True:
        k = chr(cv.waitKey(0))
        if k == 'a':
            r += 1
        elif k == 'q':
            r -= 1
        elif k == 's':
            g += 1
        elif k == 'w':
            g -= 1
        elif k == 'd':
            b += 1
        elif k == 'e':
            b -= 1
        elif k == 't':
            r += 1
            g += 1
            b += 1
        elif k == 'g':
            r -= 1
            g -= 1
            b -= 1
        elif k == 'r':
            r = 100
            g = 100
            b = 100
            cv.imshow("test_colors", img)
            continue
        elif k == 'x':
            cv.destroyAllWindows()   
            print("The RGB is ", (r, g, b))
            break
        else:
            continue
        cv.imshow("test_colors", color_filter(img, r, g, b))

if __name__=='__main__':
    img = cv.imread('/home/adityan/EE599_Deep_Learning_Project/src/data/images/2_frame374.png')
    img = cv.resize(img, (640,480))
    test_colors(img)
