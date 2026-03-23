/etc/init.d/ssh start

docker run --gpus all -itd --restart always \
-p 30030:30030 -p 1455:1455 \
--shm-size 256G \
-v /ssd1234/haozeng/workspace:/root/workspace \
-v /ssd1234/haozeng/data:/root/data \
megatron:0320