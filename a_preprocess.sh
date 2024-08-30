# -- environment setup --#
cd ~/code/research/xxx
conda activate tf
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m

# -- remove cache --#
BASE_DIR="~/code/research/xxx"
find $BASE_DIR -type d -name "__pycache__" -not -path "*/reference/*" -exec rm -r {} + # 查找并删除所有__pycache__目录，但排除reference文件夹及其子文件夹
echo "除reference文件夹外, $BASE_DIR下的所有__pycache__目录已被删除。"


# -- debug --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python -Xfrozen_modules=off tools/build_data.py  \
    --debug True

# -- experiments --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python tools/build_data.py