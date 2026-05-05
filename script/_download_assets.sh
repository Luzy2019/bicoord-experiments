# export http_proxy="http://127.0.0.1:9674"
# export https_proxy="https://127.0.0.1:9674"
# export all_proxy="socks5h://127.0.0.1:9674"

# cd assets
# python _download.py

# background_texture
unzip /home/lzy/code/BiCoord-Bench/assets/background_texture.zip -d /home/lzy/code/BiCoord-Bench/background_texture
# rm -rf background_texture.zip

# embodiments
unzip /home/lzy/code/BiCoord-Bench/assets/embodiments.zip -d /home/lzy/code/BiCoord-Bench/embodiments
# rm -rf embodiments.zip

# objects
unzip /home/lzy/code/BiCoord-Bench/assets/objects.zip -d /home/lzy/code/BiCoord-Bench/objects
# rm -rf objects.zip

cd ..
echo "Configuring Path ..."
python /home/lzy/code/BiCoord-Bench/script/update_embodiment_config_path.py