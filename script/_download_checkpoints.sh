unset all_proxy ALL_PROXY
export http_proxy="http://127.0.0.1:9674"
export https_proxy="http://127.0.0.1:9674"

python assets/_download_checkpoints_bicoord.py
