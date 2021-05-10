# sequence-to-sequence

Sequence to sequence tutorial implemented with `PyTorch`.

## Install

```sh
# Clone the project.
git clone https://github.com/ProFatXuanAll/sequence-to-sequence.git

# Install dependencies.
pipenv install
```

## Train Tokenizer

### Train Source Text Tokenizer

```sh
python run_train_tknzr.py \
character \
--exp_name 'src_tknzr' \
--dset_name 'arithmetic.src' \
--min_count 1 \
--n_vocab 20 \
--is_cased
```

### Train Target Text Tokenizer

```sh
python run_train_tknzr.py \
character \
--exp_name 'tgt_tknzr' \
--dset_name 'arithmetic.tgt' \
--min_count 1 \
--n_vocab 15 \
--is_cased
```

## Train Model

```sh
python run_train_model.py \
GRU \
--batch_size 2048 \
--ckpt_step 5000 \
--dec_d_emb 20 \
--dec_d_hid 40 \
--dec_dropout 0.0 \
--dec_n_layer 1 \
--dec_max_len 7 \
--dec_tknzr_exp 'tgt_tknzr' \
--dset_name 'arithmetic' \
--enc_d_emb 20 \
--enc_d_hid 40 \
--enc_dropout 0.0 \
--enc_n_layer 1 \
--enc_max_len 12 \
--enc_tknzr_exp 'src_tknzr' \
--epoch 100 \
--exp_name 'my_exp' \
--log_step 2500 \
--lr 5e-4 \
--max_norm 1.0 \
--seed 42
```

## Evaluate Model

```sh
python run_eval_model.py \
  --batch_size 1024 \
  --ckpt -1 \
  --dset_name 'arithmetic' \
  --exp_name 'my_exp' \
  --infr_name 'top_1'
```

## Infer Model

```sh
python run_infr_model.py \
  --ckpt -1 \
  --exp_name 'my_exp' \
  --infr_name 'top_1' \
  --src '1+1='
```
