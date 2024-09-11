import cv2


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text,clf):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        if x < 0 or y < 0 or x + w > gray_img.shape[1] or y + h > gray_img.shape[0]:
            continue

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, pred = clf.predict(gray_img[y:y + h, x:x + w])
        confidence = int(100 * (1 - pred / 250))

        if confidence > 77:

            if id == 1:
                cv2.putText(img, "Ananya", (x, y-5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

            elif id == 2 :
                cv2.putText(img, "Disha", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

            elif id == 3:
                cv2.putText(img, "Amreen", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

            elif id == 4:
                cv2.putText(img, "Prarthana", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

            elif id == 5:
                cv2.putText(img, "Pavithra", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

        else:
            cv2.putText(img, "Unknown", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 2, cv2.LINE_AA)

            coords = [x, y, w, h]
        return coords, img


def recognize(img, clf, faceCascade):
    coords = draw_boundary(img, faceCascade, 1.1, 10, (255,255,255), "face", clf)
    return img


faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read("classifier.xml")

video_capture = cv2.VideoCapture(0)
while True:
    ret, img = video_capture.read()
    img = recognize(img, clf, faceCascade)
    cv2.imshow("Face Detection",img)

    if cv2.waitKey(1) & 0xFF == 13:
        break

video_capture.release()
cv2.destroyAllWindows()