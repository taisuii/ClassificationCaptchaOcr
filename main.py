import numpy as np
from fake_useragent import UserAgent
from flask import Flask, request
import uuid
import re
import json

from loguru import logger

import requests
import time
import random

from binascii import b2a_hex, a2b_hex
import rsa
import hashlib
from Crypto.Cipher import AES
import execjs
from matplotlib import pyplot as plt

from development.predict import predict_onnx
from development.crop_image import convert_png_to_jpg


class Encrypt():
    def rsa_encrypt(self, msg):
        e = '010001'
        e = int(e, 16)
        n = '00C1E3934D1614465B33053E7F48EE4EC87B14B95EF88947713D25EECBFF7E74C7977D02DC1D9451F79DD5D1C10C29ACB6A9B4D6FB7D0A0279B6719E1772565F09AF627715919221AEF91899CAE08C0D686D748B20A3603BE2318CA6BC2B59706592A9219D0BF05C9F65023A21D2330807252AE0066D59CEEFA5F2748EA80BAB81'
        n = int(n, 16)
        pub_key = rsa.PublicKey(e=e, n=n)
        return b2a_hex(rsa.encrypt(bytes(msg.encode()), pub_key))

    def aes_encrypt(self, key, iv, content):
        def pkcs7padding(text):
            """明文使用PKCS7填充 """
            bs = 16
            length = len(text)
            bytes_length = len(text.encode('utf-8'))
            padding_size = length if (bytes_length == length) else bytes_length
            padding = bs - padding_size % bs
            padding_text = chr(padding) * padding
            self.coding = chr(padding)
            return text + padding_text

        key = key.encode('utf-8')
        iv = iv.encode('utf-8')
        """ AES加密 """
        cipher = AES.new(key, AES.MODE_CBC, iv)
        # 处理明文
        content_padding = pkcs7padding(content)
        # 加密
        encrypt_bytes = cipher.encrypt(content_padding.encode('utf-8'))
        # 重新编码
        result = b2a_hex(encrypt_bytes).decode()
        return result

    def get_random_key_16(self):
        data = ""
        for i in range(4):
            data += (format((int((1 + random.random()) * 65536) | 0), "x")[1:])
        return data

    def get_pow(self, pow_detail, captcha_id, lot_number):
        n = pow_detail['hashfunc']
        i = pow_detail['version']
        r = pow_detail['bits']
        s = pow_detail['datetime']
        o = ""
        a = r % 4
        u = r // 4
        c = '0' * u
        _ = f"{i}|{r}|{n}|{s}|{captcha_id}|{lot_number}|{o}|"
        while True:
            h = self.get_random_key_16()
            l = _ + h
            if n == "md5":
                p = hashlib.md5(l.encode()).hexdigest()
            elif n == "sha1":
                p = hashlib.sha1(l.encode()).hexdigest()
            elif n == "sha256":
                p = hashlib.sha256(l.encode()).hexdigest()

            if a == 0:
                if p.startswith(c):
                    return {"pow_msg": _ + h, "pow_sign": p}
            else:
                if p.startswith(c):
                    d = int(p[u], 16)
                    if a == 1:
                        f = 7
                    elif a == 2:
                        f = 3
                    elif a == 3:
                        f = 1
                    if d <= f:
                        return {"pow_msg": _ + h, "pow_sign": p}

    def gt_data_assembly(self, pow_detail, captcha_id, lot_number, dynamic_parameter, userresponse):
        pow_data = self.get_pow(pow_detail, captcha_id, lot_number)
        e = {
            "passtime": random.randint(1500, 4000),
            "userresponse": userresponse,
            "device_id": "",
            "lot_number": lot_number,
            "pow_msg": pow_data['pow_msg'],
            "pow_sign": pow_data['pow_sign'],
            "geetest": "captcha",
            "lang": "zh",
            "ep": "123",
            "biht": "1426265548",
            "gee_guard": '',
            "em": {"ph": 0, "cp": 0, "ek": "11", "wd": 1, "nt": 0, "si": 0, "sc": 0}
        }
        e.update(dynamic_parameter)
        e = str(e).replace('\'', '"').replace(' ', '')
        aes_key = self.get_random_key_16()
        rsa_result = str(self.rsa_encrypt(msg=aes_key), 'utf-8')
        aes_result = self.aes_encrypt(key=aes_key, iv='0000000000000000', content=e)
        w = aes_result + rsa_result
        return w


class GEETEST4():
    def __init__(self, proxies, captcha_id, risk_type):
        self.risk_type = risk_type
        if proxies == "no" or proxies == "":
            self.proxy = None
        else:
            proxies = proxies.replace("\n", "").replace("\r", "")
            self.proxy = {
                "http": f"http://{proxies}",
                "https": f"http://{proxies}"
            }
        self.captcha_id = captcha_id
        ua = UserAgent()
        self.headers = {
            "User-Agent": ua.random,
            "Referer": "https://gt4.geetest.com/"
        }
        self.session = requests.Session()
        self.session.headers = self.headers

    def get_load(self):
        url = "https://gcaptcha4.geetest.com/load"
        params = {
            "captcha_id": self.captcha_id,
            "challenge": uuid.uuid4(),
            "client_type": "web",
            "risk_type": self.risk_type,
            "lang": "zh-cn",
            "callback": "geetest_" + str(int(time.time() * 1000))
        }

        response = self.session.get(url, headers=self.headers, params=params, proxies=self.proxy).text

        response = json.loads(re.findall(r"geetest_\d+\((.*?}})\)", response)[0])
        self.load = response
        self.risk_type = response["data"]["captcha_type"]

    def get_dynamic_parameter(self):
        url = "https://gcaptcha4.geetest.com/load"
        params = {
            "captcha_id": "0b2abaab0ad3f4744ab45342a2f3d409",
            "challenge": uuid.uuid4(),
            "client_type": "web",
            "risk_type": "nine",
            "lang": "zh-cn",
            "callback": "geetest_" + str(int(time.time() * 1000))
        }
        response = requests.get(url, headers=self.headers, params=params).text
        response = json.loads(re.findall(r"geetest_\d+\((.*?}})\)", response)[0])
        static_path = response["data"]["static_path"]
        gcaptcha_js = "https://static.geetest.com/" + static_path + "/js/gcaptcha4.js"
        js = requests.get(gcaptcha_js, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36",
            "Referer": "https://gt4.geetest.com/",
        }).text.split(";Uaaco")[0]

        complete_js = """
        Uaaco = {};
        """ + js + """
        function getDynamicParameter() {
          return Uaaco.$_AL.$_HIBAt(781);
        }
        """

        dynamic_parameter = execjs.compile(complete_js).call("getDynamicParameter")
        return json.loads(dynamic_parameter)

    def get_captcha_img(self):
        url = "https://static.geetest.com/"
        try:
            bg_img_url = url + self.load["data"]["imgs"]
        except Exception as e:
            bg_img_url = url + self.load["data"]["bg"]

        bg_img = requests.get(bg_img_url, headers=self.headers).content

        img_urls = []
        try:
            img_urls = self.load["data"]["ques"]
        except:
            try:
                img_urls.append(self.load["data"]["slice"])
            except:
                img_urls = []

        if img_urls.__len__() == 0:
            return bg_img
        elif img_urls.__len__() == 1:
            return bg_img, self.session.get(url + img_urls[0], headers=self.headers).content
        else:
            img1 = self.session.get(url + img_urls[0], headers=self.headers).content
            img2 = self.session.get(url + img_urls[1], headers=self.headers).content
            img3 = self.session.get(url + img_urls[2], headers=self.headers).content
            return bg_img, [img1, img2, img3]

    def ocr(self):
        bg_img, icon_img = self.get_captcha_img()
        answer = predict_onnx(convert_png_to_jpg(icon_img), bg_img)
        return answer

    def verify(self):
        self.get_load()
        self.dynamic_parameter = self.get_dynamic_parameter()

        url = "https://gcaptcha4.geetest.com/verify"
        pow_detail = self.load["data"]["pow_detail"]
        lot_number = self.load["data"]["lot_number"]
        payload = self.load["data"]["payload"]
        process_token = self.load["data"]["process_token"]

        ocr_result = self.ocr()

        w = Encrypt().gt_data_assembly(pow_detail, self.captcha_id, lot_number, self.dynamic_parameter, ocr_result)
        params = {
            "captcha_id": self.captcha_id,
            "client_type": "web",
            "lot_number": lot_number,
            "risk_type": self.risk_type,
            "payload": payload,
            "process_token": process_token,
            "payload_protocol": "1",
            "pt": "1",
            "w": w,
            "callback": "geetest_" + str(int(time.time() * 1000))
        }
        response = self.session.get(url, headers=self.headers, params=params, proxies=self.proxy).text
        response = json.loads(re.findall(r"geetest_\d+\((.*?}})\)", response)[0])
        if response["data"]["result"] == "success":
            logger.success(json.dumps(response, ensure_ascii=False))
            return True
        else:
            logger.error(json.dumps(response, ensure_ascii=False))
            return False


app = Flask(__name__)


@app.route("/geetest4", methods=["GET", "POST"])
def geetest4():
    risk_type = "nine"
    captcha_id = "435d94a5f5b138efd5dc9f9ffc7f5621"
    proxy = ""
    Gt4 = GEETEST4(proxy, captcha_id, risk_type)
    return Gt4.verify()


def test():
    risk_type = "nine"
    captcha_id = "54088bb07d2df3c46b79f80300b0abbe"
    proxy = ""
    Gt4 = GEETEST4(proxy, captcha_id, risk_type)
    return Gt4.verify()


if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=9797)
    spendtime = []
    success = 0
    for i in range(50):
        start = time.time()
        if test():
            success = success + 1
        spendtime.append(time.time() - start)

    data = np.array(spendtime)
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(f"verify spend time, average spend: {np.mean(spendtime)} and Success rate: {str(success * 2)}/100")
    plt.ylabel("time")
    # 显示图像
    plt.show()
