# ---------------------------------------------------------------- #
# -- environment setup --#
# ---------------------------------------------------------------- #
cd /home/gulu/code/research/house_unit_match/house_unit_match_clip
conda activate tf
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m

# ---------------------------------------------------------------- #
# -- remove cache --#
# ---------------------------------------------------------------- #
BASE_DIR="/home/gulu/code/research/house_unit_match"
find $BASE_DIR -type d -name "__pycache__" -not -path "*/references/*" -exec rm -r {} + # 查找并删除所有__pycache__目录，但排除reference文件夹及其子文件夹
echo "除 references 文件夹外, $BASE_DIR 下的所有 __pycache__ 目录已被删除。"

# ---------------------------------------------------------------- #
# -- 配置accelerate --#
# ---------------------------------------------------------------- #
accelerate config default
# code /home/gulu/.cache/huggingface/accelerate/default_config.yaml

# ---------------------------------------------------------------- #
# -- 配置tmux --#
# ---------------------------------------------------------------- #
tmux new -s tcl1
tmux attach -t tcl1
exit 
tmux kill-session -t tcl1


# ---------------------------------------------------------------- #
# -- 实验 --#
# ---------------------------------------------------------------- #
cd ~/code/research/xxx
accelerate config default
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
accelerate launch tools/tune.py \
    --num_train_epochs 400 \
    --logging_steps 0.25 \
    --eval_steps 0.25 \
    --save_strategy no \
    --load_best_model_at_end False