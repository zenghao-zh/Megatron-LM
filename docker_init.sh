/etc/init.d/ssh start

# docker run --gpus all -itd --restart always \
# -p 10020:10020 -p 10021:10021 \
# --shm-size 256G \
# -v /ssd_1234/haozeng/workspace:/root/workspace \
# -v /ssd_1234/haozeng/data:/root/data \
# megatron:v1.0