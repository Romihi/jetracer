import cv2
import base64
#from jetcam.utils import bgr8_to_jpeg
import time
import flet as ft

def bgr8_to_jpeg(value, quality=75):
    return bytes(cv2.imencode('.jpg', value)[1])

#def _cv_to_base64(self,img):
def _cv_to_base64(img):
    #_, 
    #encoded = cv2.imencode(".jpg", img)
    #img_str = base64.b64encode(encoded).decode("ascii")
    img_str = base64.b64encode(img).decode("ascii")
    return img_str
    
def main(page: ft.Page):
    global image_control,i
    x, y = int(0),int(0)
    snapshot = cv2.imread('./images/0_cam_image_array_.jpg')
    snapshot = cv2.circle(snapshot, (0, 0), 8, (0, 255, 0), 3)
    retval, encoded = cv2.imencode(".jpg", snapshot)
    encoded_b = encoded.tobytes()
    image_control = base64.b64encode(encoded_b).decode("ascii")
    i = ft.Image(src_base64=image_control)
    def animate(x,y):
        x+=10
        y+=10
        print(x,y)
        #page.controls.pop()
        #snapshot = camera.value.copy()
        snapshot = cv2.imread('./images/0_cam_image_array_.jpg')
        snapshot = cv2.circle(snapshot, (x, y), 8, (0, 255, 0), 3)
        #snapshot = _cv_to_base64(snapshot)
        #snapshot_c = './images/0_cam_image_array_c.jpg'
        #cv2.imwrite(snapshot_c,snapshot)
        #e.data = base64.b64encode(snapshot)
        #e.image_control = data.decode()
        #i = ft.Image(src="https://picsum.photos/150/150", width=150, height=150)
        #i.update()
        retval, encoded = cv2.imencode(".jpg", snapshot)
        encoded_b = encoded.tobytes()
        image_control = base64.b64encode(encoded_b).decode("ascii")
        #i = ft.Image(src_base64=image_control)
        #page.clean()
        #page.add(i)
        #page.add(ft.ElevatedButton("Animate!", on_click=animate))
        page.update()

    #i = ft.Image(src="./images/0_cam_image_array_.jpg", width=224, height=224,fit=ft.ImageFit.CONTAIN)
    """i=ft.Image(
            src_base64=snapshot,
            width=480,
            height=320,
            fit=ft.ImageFit.CONTAIN,
            )   
            """
    page.add(i)
    page.add(
        ft.ElevatedButton("Animate!", on_click=animate(x,y)),
    )
    
ft.app(target=main,assets_dir="images",view=ft.WEB_BROWSER)