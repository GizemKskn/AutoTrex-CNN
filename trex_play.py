from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss
import sys

mon = {"top":457, "left":700, "width":285, "height":169}
sct = mss()

width = 125
height = 50

model = model_from_json(open("model.json","r").read())
model.load_weights("trex_weight_new.h5")

labels = ["Down", "Right", "Up"]

framerate_time = time.time()
counter = 0
i = 0
delay = 0.4


key_down_pressed = False

while True:

    img = sct.grab(mon)
    im = Image.frombytes("RGB", img.size, img.rgb)

    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255

    x = np.array([im2])
    x = x.reshape(x.shape[0], width, height, 1)
    r = model.predict(x)

    result = np.argmax(r)

    if keyboard.is_pressed('esc'):
        print("ESC pressed. Exiting...")
        sys.exit()
    
    if result == 0:
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    elif result == 2:
        
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        time.sleep(delay)

        keyboard.press(keyboard.KEY_UP)

        if i < 1175:
            time.sleep(0.5)
        elif 1175 < i and i < 5000:
            time.sleep(0.7)
        else:
            time.sleep(0.17)
        
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    

    if (time.time() - framerate_time) > 1:

        counter = 0
        framerate_time = time.time()
        if  i <= 1175:
            delay -=0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
        
        print("---------------")
        print("Down: {} \nRight: {} \n".format(r[0][0],r[0][1],r[0][2]))
        i +=1


