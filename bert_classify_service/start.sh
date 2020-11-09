docker stop bert_classify_service
docker rm -f bert_classify_service
docker rmi bert:v1
nvidia-docker build -t bert:v1 .
nvidia-docker run  --name bert_classify_service -it -p 6001:6001  -v /home/jiayao/bert_classify_service/project:/home/project bert:v1 /bin/bash
