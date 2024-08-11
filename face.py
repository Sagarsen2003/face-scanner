import cv2

a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
b = cv2.VideoCapture(0)
while True:
    c_break, d_image = b.read()
    f = cv2.cvtColor(d_image, cv2.COLOR_BGR2GRAY)
    g = a.detectMultiScale(f, 1.3, 6)

    for x1, y1, w1, h1 in g:
        cv2.rectangle(d_image, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 5)

    cv2.imshow("img", d_image)
    h = cv2.waitKey(40) & 0xFF
    if h == 20:
        break
b.release()
cv2.destroyAllWindows()
