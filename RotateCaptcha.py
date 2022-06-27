import urllib
import uuid
from io import BytesIO

from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.optimizers import SGD
import keras.backend as K
from PIL import Image


import os
import math
import cv2
import numpy as np
import requests


class RotateCaptcha():
    def __init__(self):
        # 加载模型
        model_location = os.path.join('.', 'model', 'rotnet_street_view_resnet50_keras2.hdf5')
        self.model = load_model(model_location, custom_objects={'angle_error': self.angle_error})
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=SGD(lr=0.01, momentum=0.9),
                           metrics=[self.angle_error])
        # 图像长宽尺寸，勿改
        self.size = (224, 224)

        # 下载图片使用的ua
        self.headers = {
            'Connection': 'keep-alive',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'sec-ch-ua': '"Chromium";v="88", "Google Chrome";v="88", ";Not A Brand";v="99"',
            'DNT': '1',
            'sec-ch-ua-mobile': '?0',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
            'Accept': '*/*',
            'Sec-Fetch-Site': 'same-site',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Dest': 'script',
            'Referer': 'https://www.baidu.com/s?rsv_idx=1&wd=31%E7%9C%81%E6%96%B0%E5%A2%9E%E7%A1%AE%E8%AF%8A13%E4%BE%8B+%E5%9D%87%E4%B8%BA%E5%A2%83%E5%A4%96%E8%BE%93%E5%85%A5&fenlei=256&ie=utf-8&rsv_cq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_dl=0_right_fyb_pchot_20811_01&rsv_pq=c0b53cdc0005af92&oq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_t=2452p17G6e88Hpj%2FkNppuwT%2FFjr8KeLJKT4KqqeSLqr7MhD7HbIYjtM9KVc&rsf=84b938b812815a59afcce7cc4e641b1d_1_15_8&rqid=c0b53cdc0005af92',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }

    def showImg(self, image):
        '''
        展示图片
        '''
        cv2.imshow('image', image)
        cv2.waitKey(0)

    def getImgFromDisk(self, imgPath):
        image = cv2.imread(imgPath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def getImgFromUrl(self, url):
        r = requests.get(url, headers=self.headers)
        image = Image.open(BytesIO(r.content))
        fileName = str(uuid.uuid4()) + '.' + image.format.lower()
        with open('pic/' + fileName, 'wb') as f:
            f.write(r.content)

        # image = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)  # 直接解码网络数据
        self.showImg(image)

        return image

    def predictAngle(self, image):
        diameter = image.shape[0]  # 直径
        side_length = math.floor((diameter / 2) * 1.414)  # 圆内正方形最大边长
        cropped = math.floor((diameter - side_length) / 2)
        image = image[cropped:cropped + side_length, cropped:cropped + side_length]
        image = cv2.resize(image, self.size)

        image = np.expand_dims(image, axis=0)

        x = preprocess_input(image)
        y_pred = np.argmax(self.model.predict(x), axis=1)

        return y_pred[0]

    def rotate(self, image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background

        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def angle_difference(self, x, y):
        """
        Calculate minimum difference between two angles.
        """
        return 180 - abs(abs(x - y) - 180)

    def angle_error(self, y_true, y_pred):
        """
        Calculate the mean diference between the true angles
        and the predicted angles. Each angle is represented
        as a binary vector.
        """
        diff = self.angle_difference(K.argmax(y_true), K.argmax(y_pred))
        return K.mean(K.cast(K.abs(diff), K.floatx()))
