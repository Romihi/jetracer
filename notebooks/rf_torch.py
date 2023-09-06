import psutil
import GPUtil
import gc

#メモリチェック、メモリを食い過ぎているとモデルが読み込めなくなったり、保存できずセグフォする
def checkmem():
    print("cpu mem used :",psutil.virtual_memory().percent,"%")
    GPUtil.showUtilization()
checkmem()

#Prams
##Dokeycarのモデルの読み込み
donkey = True
if donkey:
    model_name = 'models/mypilot1.pth'
    model_trt_name = 'models/mypilot_trt.pth'
    #Donkeycarで学習したckptファイルから読み込む場合
    donkey_ckpt = False
    checkpoint_path = "models/mypilot.ckpt"
    BATCH_SIZE =8
else:
    model_name = 'road_following_model.pth'
    model_trt_name = 'mypilot_trt.pth'
trtconvert = False

print("loading libraries, takes several secs")
#First, create the model. This must match the model used in the interactive training notebook.
import torch
import torchvision
device = torch.device('cuda')
CATEGORIES = ['apex']
if trtconvert:
    def create_model():    
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
        model = model.cuda().eval().half()
        print("model set")
        return model
    model = create_model()
    checkmem()

    if donkey_ckpt == True:
        ##for DK
        print("donkey car torch model used")
        from donkeycar.parts.pytorch.ResNet18 import ResNet18
        input_shape = (BATCH_SIZE, 3, 224, 224)
        model = ResNet18(input_shape=input_shape)
        model = model.cuda().eval().half()
        print("1")
        checkmem()
        model.load_from_checkpoint(checkpoint_path)
        print("2")
        checkmem()
        torch.save(model.state_dict(), model_name)
        gc.collect()
        torch.cuda.empty_cache()
        #model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(torch.load(model_name))
        model = model.cuda().eval().half()

    print("torch model used:",model_name)
    checkmem()

    #Next, load the saved model.  Enter the model path you used to save.
    #Convert and optimize the model using ``torch2trt`` for faster inference with TensorRT.  Please see the [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) readme for more details.
    print("This optimization process can take a couple minutes to complete.")
    from torch2trt import torch2trt
    data = torch.zeros((1, 3, 224, 224)).cuda().half()
    model_trt = torch2trt(model, [data], fp16_mode=True)
    print("model converted to trt")
    checkmem()

    #Save the optimized model using the cell below
    torch.save(model_trt.state_dict(), model_trt_name)
    print("model saved as ", model_trt_name)
    checkmem()
    
#Load the optimized model by executing the cell below
#import torch
from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load(model_trt_name))
print("trt model loaded :",model_trt_name)

#Create the racecar class
from jetracer.nvidia_racecar import NvidiaRacecar
car = NvidiaRacecar()
print("car program started")

#Create the camera class.
print("camera starting...")
from jetcam.csi_camera import CSICamera
camera = CSICamera(width=224, height=224, capture_fps=65)

#Finally, execute the cell below to make the racecar move forward, steering the racecar based on the x value of the apex.
#Here are some tips,
#* If the car wobbles left and right,  lower the steering gain
#* If the car misses turns,  raise the steering gain
#* If the car tends right, make the steering bias more negative (in small increments like -0.05)
#* If the car tends left, make the steering bias more postive (in small increments +0.05)
print("1")
from utils import preprocess
print("2")
import time
#import numpy as np
print("3")
import config as cfg

#STEERING_GAIN = 0.75
#STEERING_BIAS = 0.00
#car.throttle = 0.15
STEERING_GAIN = cfg.STEERING_GAIN
STEERING_BIAS = cfg.STEERING_BIAS
#throttle = 0.15
THROTTLE_GAIN = cfg.THROTTLE_GAIN
THROTTLE_BIAS = cfg.THROTTLE_BIAS
print("config loaded")

fps = 0
fps_counter = 0
last_timestamp = None
duration = 0.

print("vehicle starting in 3 secs...")
time.sleep(3)
print("Go!")

last_timestamp = time.time()
while True:
    image = camera.read()
    image = preprocess(image).half()
    output = model_trt(image).detach().cpu().numpy().flatten()
    x = float(output[0])
    y = float(output[1])
    car.steering = x * STEERING_GAIN + STEERING_BIAS
    car.throttle = y * THROTTLE_GAIN + THROTTLE_BIAS

    if time.time() - last_timestamp > 1:
        fps = fps_counter
        fps_counter = 0
        last_timestamp = time.time()
        print("current fps :",fps)
    else:
        fps_counter += 1