#Prams
model_name = 'road_following_model.pth'
model_trt_name = 'road_following_model_trt.pth'
donkey = False

print("loading libraries, takes several secs")
#First, create the model. This must match the model used in the interactive training notebook.
import torch
import torchvision

CATEGORIES = ['apex']
device = torch.device('cuda')
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2 * len(CATEGORIES))
model = model.cuda().eval().half()
print("model set")

print("torch model loading and trt converting on.\n It may fall into Segmentation fault.\n In this case, please set trtconvert = False")
#Next, load the saved model.  Enter the model path you used to save.
##for DK
#donkeycar_path='C:\Users\hkior\projects\donkeycar\jethalowinnano\'
#from donkeycar_path import myconfig
if donkey == True:
    print("donkey car torch model used")
    from donkeycar.parts.pytorch.ResNet18 import ResNet18
    BATCH_SIZE =32
    input_shape = (BATCH_SIZE, 3, 224, 224)
    model = ResNet18(input_shape=input_shape)
    checkpoint_path = "models/mypilot.ckpt"
    checkpoint = torch.load(checkpoint_path)
    #model.load_from_checkpoint(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])

else:
    model.load_state_dict(torch.load(model_name))

print("torch model loaded")

#Convert and optimize the model using ``torch2trt`` for faster inference with TensorRT.  Please see the [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) readme for more details.
#This optimization process can take a couple minutes to complete. 
from torch2trt import torch2trt
data = torch.zeros((1, 3, 224, 224)).cuda().half()
model_trt = torch2trt(model, [data], fp16_mode=True)
print("model converted to trt")

#Save the optimized model using the cell below
torch.save(model_trt.state_dict(), )
print("model saved as ", model_trt_name)
