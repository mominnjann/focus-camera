import cv2

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask = fgbg.apply(gray)
    # Clean mask
    fgmask = cv2.medianBlur(fgmask, 5)
    thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion = False
    for c in contours:
        if cv2.contourArea(c) > 800:  # tune
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            motion = True

    cv2.putText(frame, f"Motion: {motion}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if motion else (0,0,255), 2)
    cv2.imshow("Motion", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()