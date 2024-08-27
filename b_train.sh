# -- environment setup --#
cd ~/code/research/xxx
conda activate tf
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m

# -- remove cache --#
BASE_DIR="~/code/research/xxx"
find $BASE_DIR -type d -name "__pycache__" -exec rm -r {} + # Find and remove all __pycache__ directories
echo "All __pycache__ directories under $BASE_DIR have been removed."

# -- tmux setup --#
tmux new -s tcl1
tmux attach -t tcl1
exit 
tmux kill-session -t tcl1

# -- debug --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python -Xfrozen_modules=off tools/train_net.py \
    --mydebug True

# -- experiments --#
free -m && sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches' && free -m
python tools/train_net.py \
    --experiment_name "experiment--batch_size_4096" \
    --max_epochs 20_000 --resume  "epoch_5000.pt"\
    --batch_size 4096