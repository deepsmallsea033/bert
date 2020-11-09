
FROM tensorflow/tensorflow:1.13.1-gpu-py3

ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update  && apt-get install -y vim
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.18.1  pandas==0.24.2 Flask==1.1.2 Flask-Cors==3.0.8 tornado==6.0.4

CMD /bin/bash
ENV LANG=C.UTF-8
CMD bash

