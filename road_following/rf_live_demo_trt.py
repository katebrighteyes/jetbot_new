gst_str = ("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)224, height=(int)224, format=(string)NV12, framerate=(fraction)60/1 ! nvvidconv flip-method=0 ! video/x-raw, width=(int)224, height=(int)360, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")

video_path = '../record/line.mp4'
import cv2
import numpy as np
from Adafruit_MotorHAT import Adafruit_MotorHAT

import torchvision
import torch

import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

from torch2trt import TRTModule 


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]
	
angle = 0.0
angle_last = 0.0

#speed_gain_value = 1.0
speed_gain_value = 0.18
steering_value = 0.0001
#steering_gain_value = 1.0
#steering_dgain_value = 0.03
steering_gain_value = 0.03
steering_dgain_value = 0.0
steering_bias_value = 0.0
def executeModel(image):
    global angle, angle_last

    xy = model(preprocess(image)).detach().float().cpu().numpy().flatten()
    x = xy[0]
    y = (0.5 - xy[1]) / 2.0

    print(x)
    print(y)
    print('---')

    
    speed_value = speed_gain_value

    angle = np.arctan2(x, y)
    pid = angle * steering_gain_value + (angle - angle_last) * steering_dgain_value
    angle_last = angle

    #global steering_value
    #print(steering_value)
    steering_value = pid + steering_bias_value
 
    
    left_motor_value = max(min(speed_value + steering_value, 1.0), 0.0)
    right_motor_value = max(min(speed_value - steering_value, 1.0), 0.0)	
   
    set_speed(motor_left_ID,   left_motor_value)
    set_speed(motor_right_ID,  right_motor_value)

def set_speed(motor_ID, value):
	max_pwm = 115.0
	speed = int(min(max(abs(value * max_pwm), 0), max_pwm))

	if motor_ID == 1:
		motor = motor_left
	elif motor_ID == 2:
		motor = motor_right
	else:
		return
	
	motor.setSpeed(speed)

	if value > 0:
		motor.run(Adafruit_MotorHAT.FORWARD)
	else:
		motor.run(Adafruit_MotorHAT.BACKWARD)


# stops all motors
def all_stop():
	motor_left.setSpeed(0)
	motor_right.setSpeed(0)

	motor_left.run(Adafruit_MotorHAT.RELEASE)
	motor_right.run(Adafruit_MotorHAT.RELEASE)

def imageCopy(src):
    return np.copy(src)
	
def imageProcessing(output):
    executeModel(output)
    return output

def Video(openpath, savepath = None):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(width)
    print(height)
    print('--')

    out = None
    #if savepath is not None:
        #out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    #cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    linecolor1 = (0,240,240)
    linecolor2 = (230,0,0)

    try:
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                # Our operations on the frame come here

                frame = cv2.resize(frame, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                output = imageProcessing(frame)
                #frame = np.copy(frame)			
                #im = cv2.line(im, (112, 0), (112, 224), linecolor1, 5, cv2.LINE_AA)			

                cv2.imshow("Input", frame)			

            else:
                break
            # waitKey(int(1000.0/fps)) for matching fps of video
            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:  
        print("key int")
        all_stop()
        cap.release()
        cv2.destroyAllWindows()
        return

    # When everything done, release the capture
    cap.release()
    if out is not None:
        out.release()
    all_stop()
    cv2.destroyAllWindows()
    return
   
#if __name__=="__main__":
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)

model_trt = TRTModule()
model_trt.load_state_dict(torch.load('best_model_xy_trt.pth'))

device = torch.device('cuda')
model = model_trt.to(device)
model = model_trt.eval()

motor_driver = Adafruit_MotorHAT(i2c_bus=1)

motor_left_ID = 1
motor_right_ID = 2

motor_left = motor_driver.getMotor(motor_left_ID)
motor_right = motor_driver.getMotor(motor_right_ID)


'''
speed_value = speed_gain_value

angle = 0
pid = angle * steering_value + (angle - angle_last) * steering_dgain_value
angle_last = angle
    
steering_value = pid + steering_bias_value
print(steering_value)
'''
print("motor_driver ready")

Video(gst_str)
#robot.stop()
