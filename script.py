import cv2, time, pandas
from datetime import datetime

first_frame = None
status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["start", "end"])

# webcam index, maybe you have an external camera, 
# then it will have an index of 1
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # to remove noise and increase accuracy, 
    # look up gaussian blurring documentary
    gray = cv2.GaussianBlur(gray, (21,21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame, gray)

    # 30 for threshold limit, 
    # 255 for white color (assign this when > 30), 
    # and then a threshold method
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    # to smooth the image
    # we can pass in the kernel array in the second parameter (more complicated)
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # use copy() don't want to modify the original image
    # external contour
    # approximation method
    (cnts,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        status = 1

        (x, y, width, height) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+width, y+height), (0,255,0), 3)

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        if status == 1:
            times.append(datetime.now())
        break

# with the step of 2 because a pair of start/ end time
for i in range(0, len(times), 2):
    df = df.append({"start": times[i], "end":times[i+1]}, ignore_index=True)

df.to_csv("Times.csv")

video.release()
cv2.destroyAllWindows()