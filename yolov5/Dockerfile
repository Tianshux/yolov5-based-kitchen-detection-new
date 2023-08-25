#基于的基础镜像
FROM python:3.8.2
#解决 ImportError: libGL.so.1: cannot open shared object file: No such file or directory 问题
#RUN (apt-get update) && (apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6)
#ENV PYTHONUNBUFFERED 1
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list \
    && sed -i 's/security.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list

RUN apt-get update
RUN apt-get install libgl1-mesa-glx -y
# 设置code文件夹是工作目录
WORKDIR /code
#代码添加到code文件夹
COPY . /code
# 安装支持
RUN pip install --upgrade pip
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host pypi.tuna.tsinghua.edu.cn

CMD ["python","detect_all.py"]

