import os
import cv2
import asyncio
import numpy as np
import paddlehub as hub
import requests
import base64
from PIL import Image


from wechaty import (
    Contact,
    FileBox,
    Message,
    Wechaty,
    ScanStatus,
)

module = hub.Module(name="deeplabv3p_xception65_humanseg")

def img_transform(img_path, img_name):
    """
    将人像抠出来
    img_path: 图片的路径
    img_name: 图片的文件名
    """
    # 图片转换后存放的路径
    

    # 模型预测
    
    results = module.segmentation(images=[cv2.imread(img_path)],visualization=True,output_dir="./")

    # 将图片保存到指定路径
    
    img_new_path=results[0]['save_path']
    # 返回新图片的路径
    return img_new_path

def blend_images(fore_image, base_image):
    """
    将抠出的人物图像换背景
    fore_image: 前景图片，抠出的人物图片
    base_image: 背景图片
    """
    # 读入图片
    base_image = Image.open(base_image).convert('RGB')
    fore_image = Image.open(fore_image).resize(base_image.size)

    # 图片加权合成
    scope_map = np.array(fore_image)[:,:,-1] / 255
    scope_map = scope_map[:,:,np.newaxis]
    scope_map = np.repeat(scope_map, repeats=3, axis=2)
    res_image = np.multiply(scope_map, np.array(fore_image)[:,:,:3]) + np.multiply((1-scope_map), np.array(base_image))
    
    #保存图片
    res_image = Image.fromarray(np.uint8(res_image))
    res_image.save("./image-new/blend_res_img.jpg")
    return
    
    
dict1 = { '水星':'水星,中国古称辰星，西汉《史记‧天官书》的作者司马迁从实际观测发现辰星呈灰色，与五行学说联系在一起，以黑色属水，将其命名为水星。水星是太阳系的八大行星中最小且最靠近太阳的行星。轨道周期是87.9691天，116天左右与地球会合一次，公转速度远远超过太阳系的其它行星。水星是表面昼夜温差最大的行星，大气层极为稀薄无法有效保存热量，白天时赤道地区温度可达432°C，夜间可降至-172°C。水星的轴倾斜是太阳系所有行星中最小的（大约1⁄30度），但有最大的轨道偏心率。水星在远日点的距离大约是在近日点的1.5倍。','金星':'金星,司马迁从实际观测发现太白为白色，与“五行”学说联系在一起，正式把它命名为金星。金星在夜空中的亮度仅次于月球，是第二亮的天体，视星等可以达到-4.7等，足以在地面照射出影子。由于金星是在地球内侧的内行星，它永远不会远离太阳运行：它的离日度最大值为47.8°。金星是一颗与地球相似的类地行星，常被称为地球的姊妹星。它有着四颗类地行星中最浓厚的大气层，其中超过96%都是二氧化碳，金星表面的大气压力是地球的92倍。其表面的平均温度高达735 K（462 °C），是太阳系中最热的行星，比最靠近太阳的水星还要热。','地球':'我们可爱的蓝星，太阳系由内及外的第三颗行星，也是太阳系中直径、质量和密度最大的类地行星，距离太阳约1.496亿千米（1天文单位）。地球自西向东自转，同时围绕太阳公转。现有45.5亿岁，有一个天然卫星——月球，二者组成一个天体系统——地月系统。45.5亿年以前起源于原始太阳星云，地球赤道半径6378.137千米，极半径6356.752千米，平均半径约6371千米，赤道周长约为40075千米，呈两极稍扁赤道略鼓的不规则的椭球体。地球表面积5.1亿平方千米，其中71%为海洋，29%为陆地，在太空上看地球总体上呈蓝色。大气层，主要成分为氮气和氧气以及少量二氧化碳、氩气等。','火星':'其橘红色外表是因为地表被赤铁矿（氧化铁）覆盖，火星的直径约为地球的一半，自转轴倾角、自转周期则与地球相近，但公转周期是地球的两倍。火星亮度最高可达-2.9等，但在大部分时间里比木星暗。2021年5月我国的“祝融号”火星车和天问一号探测器成功登陆火星！这也是我国首次将探测器成功登陆火星','木星':'木星,司马迁从实际观测发现岁星呈青色，与“五行”学说联系在一起，正式把它命名为木星,木星是颗巨行星，质量是太阳的千分之一，但却是太阳系其他行星质量总和的2.5倍。木星的主要成分是氢，但只占十分之一分子数量的氦，却占了总质量的四分之一；它可能有岩石核心和重元素，但没有可以明确界定的固体表面。由于快速地自转，木星的外观呈现扁球体。大气层依纬度成不同的区与带，在彼此的交界处有湍流和风暴作用着。最显著的例子就是大红斑，这是17世纪第一次被望远镜见到后就未曾停歇过的巨大风暴。环绕着木星的还有微弱的行星环和强大的磁层，包括4颗1610年发现的伽利略卫星，至2019年12月已经发现79颗卫星。木卫三是其中最大的一颗，其直径大于行星中的水星。','土星':'土星，中国古代人根据五行学说结合肉眼观测到的土星的颜色（黄色）来命名的，亦称之为镇星（常写作填星）。土星是气态巨行星，主要由氢组成，还有少量的氦与少量元素，内部的核心包括岩石和冰，外围由数层金属氢和气体包覆着。最外层的大气层在外观上通常情况下都是平淡的，虽然有时会有长时间存在的特征出现。土星的风速高达1800千米/时，风速明显比木星快。土星的行星磁场强度介于地球和更强的木星之间。土星有一个显著的行星环系统，主要的成分是冰的微粒和较少数的岩石残骸以及尘土。已经确认的土星的卫星有82颗，是八大行星中最多。其中，土卫六是土星系统中最大和太阳系中第二大的卫星，仅次于木卫三，比行星中的水星还要大，并且土卫六是太阳系仅有的拥有明显大气层的卫星。土星自转一周等于10小时33分38秒，大约是地球的半天时长。','天王星':'天王星，天王星是第一颗使用望远镜发现的行星。威廉·赫歇尔在1781年3月13日于自宅庭院中发现了这颗行星。天王星和海王星的内部和大气构成和更巨大的气态巨行星（木星、土星）不同，天文学家设立了冰巨星分类来定义它们。天王星拥有27颗已知天然卫星，其中有5颗规模较大，另外还有13条较为暗弱的行星环。','海王星':'海王星，海王星的大气层的化学组成以氢分子和氦为主。此外，海王星大气中还有微量的甲烷，这是使行星呈蓝色的原因之一。海王星有着强烈的风暴，测量到的风速高达2400km/h。海王星云顶温度是-218摄氏度（55K），比天王星云顶温度稍高。据推测，海王星很可能有一个炽热的内部，其核心的温度约7000℃，和大多数已知的行星相似。海王星的质量稍大于天王星，密度稍大于天王星，而半径稍小于天王星。'}


async def on_message(msg: Message):
    talker = msg.talker()

    if msg.text() == 'ding':
        await talker.say('这是自动回复: dong dong dong')

    if msg.text() == '水星':
        await talker.say(dict1['水星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAs1Ig.jpg'
        file_box_01 = FileBox.from_url(url=url, name='shui.jpg')
        await msg.say(file_box_01)

    if msg.text() == '金星':
        await talker.say(dict1['金星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAO2ZT.jpg'
        file_box_02 = FileBox.from_url(url=url, name='jin.jpg')
        await msg.say(file_box_02)

    if msg.text() == '地球':
        await talker.say(dict1['地球'])
        url = 'https://z3.ax1x.com/2021/08/04/fAXpQI.jpg'
        file_box_03 = FileBox.from_url(url=url, name='di.jpg')
        await msg.say(file_box_03)

    if msg.text() == '火星':
        await talker.say(dict1['火星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAjmND.jpg'
        file_box_04 = FileBox.from_url(url=url, name='huo.jpg')
        await msg.say(file_box_04)

    if msg.text() == '木星':
        await talker.say(dict1['木星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAjX2d.jpg'
        file_box_05 = FileBox.from_url(url=url, name='mu.jpg')
        await msg.say(file_box_05)

    if msg.text() == '土星':
        await talker.say(dict1['土星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAvPIS.jpg'
        file_box_06 = FileBox.from_url(url=url, name='mu.jpg')
        await msg.say(file_box_06)

    if msg.text() == '天王星':
        await talker.say(dict1['天王星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAvJMR.png'
        file_box_07 = FileBox.from_url(url=url, name='tian.png')
        await msg.say(file_box_07)

    if msg.text() == '海王星':
        await talker.say(dict1['海王星'])
        url = 'https://z3.ax1x.com/2021/08/04/fAvHLq.jpg'
        file_box_08 = FileBox.from_url(url=url, name='hai.jpg')
        await msg.say(file_box_08)




    if msg.text() == 'hi' or msg.text() == '你好' or msg.text() == '@徐嫣琼':
        await talker.say('你好，欢迎来到太空漫游指南1.0: 机器人目前的功能是\n- 收到八大星球的名称, 自动回复这个星球的介绍与图片\n- 收到"你的照片", 自动合成一张你的宇宙漫游照')

    if msg.text() == '图片':
        url = 'https://z3.ax1x.com/2021/08/04/fFcS2R.jpg'

        # 构建一个FileBox
        file_box_1 = FileBox.from_url(url=url, name='xx.jpg')

        await msg.say(file_box_1)
    
    if msg.type() == Message.Type.MESSAGE_TYPE_IMAGE:
        await talker.say('正在为您合成图片，请稍后')
        # 将Message转换为FileBox
        file_box_2 = await msg.to_file_box()

        # 获取图片名
        img_name = file_box_2.name

        # 图片保存的路径
        img_path = './image/' + img_name

        # 将图片保存为本地文件
        await file_box_2.to_file(file_path=img_path)

        # 调用图片风格转换的函数
        img_new_path = img_transform(img_path, img_name)
        blend_images(img_new_path, 'zhurong.jpg')
        

        # 从新的路径获取图片
        file_box_3 = FileBox.from_file("./image-new/blend_res_img.jpg")
        
        await msg.say(file_box_3)


async def on_scan(
        qrcode: str,
        status: ScanStatus,
        _data,
):
    print('Status: ' + str(status))
    print('View QR Code Online: https://wechaty.js.org/qrcode/' + qrcode)


async def on_login(user: Contact):
    print(user)


async def main():
    # 确保我们在环境变量中设置了WECHATY_PUPPET_SERVICE_TOKEN
    if 'WECHATY_PUPPET_SERVICE_TOKEN' not in os.environ:
        print('''
            Error: WECHATY_PUPPET_SERVICE_TOKEN is not found in the environment variables
            You need a TOKEN to run the Python Wechaty. Please goto our README for details
            https://github.com/wechaty/python-wechaty-getting-started/#wechaty_puppet_service_token
        ''')

    bot = Wechaty()

    bot.on('scan', on_scan)
    bot.on('login', on_login)
    bot.on('message', on_message)

    await bot.start()

    print('[Python Wechaty] Ding Dong Bot started.')


asyncio.run(main())
