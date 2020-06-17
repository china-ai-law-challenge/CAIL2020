date
source ~/.bashrc
export PATH="/yrfs1/rc/zpchen/tools/anaconda9/bin:$PATH"
bert_dir='/ps2/rc/zpchen/distillation/RoBERTa_base/pt_base_model/'
python run_cail.py \
    --name train_v1 \
    --bert_model $bert_dir \
    --data_dir cail20_v1 \
    --batch_size 2 \
    --eval_batch_size 32 \
    --lr 1e-5 \
    --gradient_accumulation_steps 1 \
    --epochs 10 \

date
