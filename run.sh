pip install --upgrade pip

pip install wechaty==0.8.15
pip install wechaty-grpc==0.20.19
pip install wechaty-puppet==0.3dev10
pip install wechaty-puppet-service==0.8.1


#hub install animegan_v2_shinkai_33

hub install deeplabv3p_xception65_humanseg

# 设置环境变量
export WECHATY_PUPPET_SERVICE_TOKEN=puppet_padlocal_c2fc312912f6449cada8ddd8562c70ef
export WECHATY_PUPPET_HOSTIE_TOKEN=puppet_padlocal_c2fc312912f6449cada8ddd8562c70ef

# 设置使用GPU进行模型预测
export CUDA_VISIBLE_DEVICES=0

# 创建两个保存图片的文件夹
mkdir -p image
mkdir -p image-new
mkdir -p image1
mkdir -p image-new1


# 运行python文件
python run.py
