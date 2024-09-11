import cv2


def generate_data(img, id, img_id):
    cv2.imwrite("data/user"+"."+str(id)+".jpg", img)


def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_img,scaleFactor,minNeighbors)
    coords = []
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x,y) ,(x+w,y+h), color, 2)
        cv2.putText(img, text, (x, y-4), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1, cv2.LINE_AA)
        coords = [x, y, w, h]
    return coords, img


def detect(img, facecascade, img_id):
    color = {"Blue" : (255, 0, 0), "Red": (0, 0, 255), "Green": (0, 255, 0), "Yellow": (255, 255, 0)}
    coords, img = draw_boundary(img ,facecascade ,1.1 ,10 , color["Yellow"] ,"Face")
    if len(coords) == 4:
        roi_img = img[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2]]
        user_id = 4

        generate_data(roi_img, user_id, img_id)

    return img


facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


video_cap = cv2.VideoCapture(0)
img_id = 0
while True:
    ret, img = video_cap.read()
    img = detect(img, facecascade, img_id)
    cv2.imshow("Face Detection", img)
    img_id = img_id + 1
    if cv2.waitKey(10) == ord('q'):
        break
video_cap.release()
cv2.destroyAllWindows()