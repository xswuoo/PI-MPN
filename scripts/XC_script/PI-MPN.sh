export CUDA_VISIBLE_DEVICES=1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/xc-lpr/ \
  --data_path od.npy \
  --model_id XC_5_1 \
  --model PI-MPN \
  --data custom \
  --seq_len 5 \
  --pred_len 1 \
  --n_layers 1 \
  --num_nodes 333 \
  --num_pairs 378 \
  --num_categories 499 \
  --train_epochs 100 \
  --batch_size 4 \
  --patience 20 \
  --itr 1