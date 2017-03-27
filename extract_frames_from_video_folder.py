"""
Class to read/stream images off the PiCamera.
"""
import time
#from picamera import PiCamera
import cv2
import os
"""
def capture_images(save_folder):
    #Stream images off the camera and save them.
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 5

    # Warmup...
    time.sleep(2)

    # And capture continuously forever.
    for _ in camera.capture_continuous(
            save_folder + '{timestamp}.jpg',
            'jpeg', use_video_port=True
    ):
        pass

if __name__ == '__main__':
    capture_images('/mnt/usbstorage/images/')
"""
def get_frame(category):
    for cat in category:
        count = 0
        for file in os.listdir("./"+cat):
            cam = cv2.VideoCapture("./" + cat + "/" + file)
            print "./" + cat + "/" + file
            #success = True
            success, frame = cam.read()
            #frame1 = cv2.resize(frame, (640, 480))

            while success:
                success, frame = cam.read()
                if success:
                    frame1 = cv2.resize(frame, (640, 480))
                    #print 'Read a new frame: ', success
                    cv2.imwrite("./images/"+"classifications/"+cat+"/frame%d.jpg" % count, frame1)  # save frame as JPEG file
                    count += 1
                else:
                    break



if __name__ == "__main__":
    get_frame(['carcrash', 'fight', 'gun'])


