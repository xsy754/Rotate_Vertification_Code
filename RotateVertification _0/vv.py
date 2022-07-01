from PIL import Image
import numpy as np
import Rotatecapcha

def transparent_back(img):
    img = img.convert('RGBA')
    L, H = img.size
    color_0 = img.getpixel((2,2))
    for h in range(H):
        for l in range(L):
            dot = (l,h)
            color_1 = img.getpixel(dot)
            if color_1 ==(255,255,255,255):
                color_1 = color_1[:-1] + (0,)
                img.putpixel(dot,(0,0,0,0))
    return img



image=Image.open(r"D:\lqbz\rotate\25.jpg")
img=transparent_back(image)
corrected_image = img.rotate(-169.2)
Image._show(corrected_image)
corrected_image.save('c30.png')