# -- environment setup --#
cd /home/gulu/code/research/house_unit_match/house_unit_match_clip
conda activate tf
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m

# -- remove cache --#
BASE_DIR="/home/gulu/code/research/house_unit_match"
find $BASE_DIR -type d -name "__pycache__" -exec rm -r {} + # Find and remove all __pycache__ directories
echo "All __pycache__ directories under $BASE_DIR have been removed."

# -- debug --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python -Xfrozen_modules=off tools/build_data.py  \
    --debug True

# -- experiments --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python tools/build_data.py