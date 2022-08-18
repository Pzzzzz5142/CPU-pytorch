import requests
import time

headers = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://neu.niutrans.com",
    "Accept-Encoding": "gzip, deflate, br",
    "Host": "test.niutrans.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
    "Accept-Language": "zh-CN,zh-Hans;q=0.9",
    "Referer": "https://neu.niutrans.com/",
    "Connection": "keep-alive",
}
t = int(time.time() * 1000)
params = {"src_text": "你好", "source": "text", "t": t, "time": t + 3}

re = requests.get(
    "https://test.niutrans.com/NiuTransServer/language", headers=headers, params=params
)

lang = re.json()["language"]

trans_params = {"tgt"}
