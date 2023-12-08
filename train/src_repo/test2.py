from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import requests
from PIL import Image
from io import BytesIO
from selenium.webdriver.common.by import By
import os
import re


target_url = "https://yidong-32180.cvmart.net:32180/yidong-tensorboard-instance-43565-93f3adefc360e7f509a188363ba52b97/#text"

# 禁用图片加载
options = webdriver.EdgeOptions()

# Set preferences to block image loading
prefs = {
    "profile.managed_default_content_settings.images": 2  # 2 blocks image loading
}
mobile_user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Mobile Safari/537.36"

options.add_experimental_option("prefs", prefs)
options.add_argument(f'user-agent={mobile_user_agent}')
driver = webdriver.Edge(options=options)
driver.get(target_url)

a=input('input anything to continue')
print('done')


