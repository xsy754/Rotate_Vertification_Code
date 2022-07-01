import re
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
import cv2
import numpy as np
import requests
from PIL import Image
import Rotatecapcha
'''
selenium进入网址，对图片进行解析需要旋转的角度，返回移动长度，完成验证码
'''

driver = webdriver.Edge(r'C:\Users\PC\environment_new\Scripts\msedgedriver.exe')
cookies = {
    'BIDUPSID': '0149403893FB17D9868DDC00E811FCE7',
    'PSTM': '1615095665',
    'BAIDUID': '0149403893FB17D9C00F7D3EFEEAD014:FG=1',
    'delPer': '0',
    'PSINO': '2',
    'H_PS_PSSID': '33257_33273_31660_33594_33570_33591_26350_33265',
    'BA_HECTOR': '040g2l8k2g0l218k311g48prj0r',
    'BDORZ': 'B490B5EBF6F3CD402E515D22BCDA1598',
    'BAIDUID_BFESS': '0149403893FB17D9C00F7D3EFEEAD014:FG=1',
    'HOSUPPORT': '1',
    'HOSUPPORT_BFESS': '1',
    'pplogid': '6942ne4CqAM25cfC4EssHF4KPYKQwU5eZfkq2hxkeydl8lc3jlmZya3gKkwSeRGtnrmryRCGksGhdH6vvVel%2FMHsiFWkNBqc0k%2B5oLijPesWvVA%3D',
    'pplogid_BFESS': '6942ne4CqAM25cfC4EssHF4KPYKQwU5eZfkq2hxkeydl8lc3jlmZya3gKkwSeRGtnrmryRCGksGhdH6vvVel%2FMHsiFWkNBqc0k%2B5oLijPesWvVA%3D',
    'UBI': 'fi_PncwhpxZ%7ETaJcwoQzv%7Etk1GbYFHnDEF1qmAF2zK6dHasFcbPclSCm%7En-w%7ENK7VbkDIllyhZUKHBMdm5g',
    'UBI_BFESS': 'fi_PncwhpxZ%7ETaJcwoQzv%7Etk1GbYFHnDEF1qmAF2zK6dHasFcbPclSCm%7En-w%7ENK7VbkDIllyhZUKHBMdm5g',
    'logTraceID': 'dc5efb79487f49534f9e47dfe497392b33c9992e86bd78d6f0',
}
headers = {
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
   #'Referer': 'https://www.baidu.com/s?rsv_idx=1&wd=31%E7%9C%81%E6%96%B0%E5%A2%9E%E7%A1%AE%E8%AF%8A13%E4%BE%8B+%E5%9D%87%E4%B8%BA%E5%A2%83%E5%A4%96%E8%BE%93%E5%85%A5&fenlei=256&ie=utf-8&rsv_cq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_dl=0_right_fyb_pchot_20811_01&rsv_pq=c0b53cdc0005af92&oq=np.random.choice+%E4%B8%8D%E9%87%8D%E5%A4%8D&rsv_t=2452p17G6e88Hpj%2FkNppuwT%2FFjr8KeLJKT4KqqeSLqr7MhD7HbIYjtM9KVc&rsf=84b938b812815a59afcce7cc4e641b1d_1_15_8&rqid=c0b53cdc0005af92',
    'Accept-Language': 'zh-CN,zh;q=0.9',
}
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
def distance(angle):
    dis=(212/360)*angle
    return dis
def GetVerPic():
    driver.get(
        'https://wappass.baidu.com/static/captcha/tuxing.html?ak=572be823e2f50ea759a616c060d6b9f1&backurl=https%3A%2F%2Fmbd.baidu.com%2Fnewspage%2Fdata%2Flandingsuper%3Fthird%3Dbaijiahao%26baijiahao_id%3D1724870502967348866%26wfr%3Dspider%26c_source%3Dkunlun&timestamp=1656035427&signature=490693a3db0ce11a781b08259c7842c6')
    pic_url = WebDriverWait(driver, 20).until(EC.visibility_of_element_located(
        (By.XPATH, "//img[@class='vcode-spin-img']"))).get_attribute("src") + ".jpg"
    print('1')
    resp = requests.get(pic_url, headers=headers,
                        cookies=cookies)
    pic = resp.content
    image = cv2.imdecode(np.frombuffer(pic, np.uint8), cv2.IMREAD_COLOR)
    print('2')
    # image=Image.fromarray(image)
    # box=(50,50,300,300)
    # image=image.crop(box)
    # Image._show(image)
    with open("D:\\lqbz\\rotate\\{}.jpg".format(int(time.time())), "wb") as f:
        f.write(pic)
    return image

def main():
    VerPic=GetVerPic()
    # image=transparent_back(VerPic)
    rotateCaptcha=Rotatecapcha.RotateCaptcha()
    predicted_angle = rotateCaptcha.predictAngle(image=VerPic) # 预测的旋转角度
    print("需旋转角度：{}".format(predicted_angle))
    corrected_image = rotateCaptcha.rotate(VerPic, -predicted_angle)  # 矫正后图像
    rotateCaptcha.showImg(corrected_image)  # 展示矫正后图像
    move_distance=distance(predicted_angle) # 滑块移动距离
    print(move_distance)
    # 定位到滑块
    ele = driver.find_element_by_xpath(
        '//html/body/div[4]/div[1]/div/div[2]/div[2]')
    # 实例化对象
    action = ActionChains(driver)
    # 拖动滑块
    time.sleep(1)
    action.drag_and_drop_by_offset(ele, xoffset=move_distance, yoffset=0).perform()


if __name__ == '__main__':
    main()
