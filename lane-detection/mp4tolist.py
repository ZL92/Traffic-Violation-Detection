import cv2
def getFrame(videoPath, svPath):
    cap = cv2.VideoCapture(videoPath)
    numFrame = 0
    list_file = svPath + '/test.txt'
    f=open(list_file, 'w')
    while numFrame<300:
        numFrame += 1
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            else:
                # cv2.imshow('video', frame)
                newPath = svPath + str(numFrame) + ".jpg"
                newPath_to_list = '/'+str(numFrame)+".jpg"+'\n'
                print("numFrame {} text {}".format(numFrame,newPath_to_list))
                f.write(newPath_to_list)
                frame = cv2.resize(frame, (1280,720), interpolation=cv2.INTER_CUBIC)
                cv2.imencode('.jpg', frame)[1].tofile(newPath)
        # if cv2.waitKey(10) == 27:
        #     break
    f.close()
 
getFrame('/home/gym/Ultra-Fast-Lane-Detection/90.mp4','/home/gym/Ultra-Fast-Lane-Detection/90/')
