from detect import *
import cv2
import os

# clear gpu
# torch.cuda.empty_cache()

y5_model = Y5Detect(weights="./weight/best_2.pt")



def Display_memory():
    total_memory, used_memory_before, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    print ("\033[A                             \033[A")
    print("    Memory: %0.2f GB / %0.2f GB"%(used_memory_before/1024, total_memory/1024))



source = 0
# source = '/home/thien/Videos/video2.mp4'
source = '/home/thien/Downloads/test.MOV'
cap = cv2.VideoCapture(source)

while True:
    # Display_memory()
    # Capture frame-by-frame
    ret, image_bgr = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bbox, label, score = y5_model.predict(image)
    for i in range(len(label)):
        # if label[i] == 'head':
        image, _ = draw_boxes(image_bgr, bbox[i],label = label[i] ,scores=score[i])

    cv2.imshow('person', image_bgr)

    k = cv2.waitKey(1) & 0xff
    if k == ord('q') or k == 27:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()