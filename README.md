# sequence-to-sequence

Sequence to sequence tutorial implemented with `PyTorch`.

## Install

```sh
# Clone the project.
git clone https://github.com/ProFatXuanAll/sequence-to-sequence.git

# Install dependencies.
pipenv install
```

## Train

```sh
python run_train_model.py RNN \
  --ckpt 0 \
  --ckpt_step 1000 \
  --dec_d_emb 100 \
  --dec_d_hid 300 \
  --dec_dropout 0.1 \
  --dec_is_cased \
  --dec_n_layer 2 \
  --dec_n_vocab 10000 \
  --dec_pad_id 0 \
  --enc_d_emb 100 \
  --enc_d_hid 300 \
  --enc_dropout 0.1 \
  --enc_is_cased \
  --enc_n_layer 2 \
  --enc_n_vocab 10000 \
  --enc_pad_id 0 \
  --exp_name 'test' \
  --is_bidir \
  --log_step 500
```
