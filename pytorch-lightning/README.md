# PyTorch-lightning

A PyTorch Wrapper.

https://github.com/PyTorchLightning/pytorch-lightning

## SYNOPSIS

```bash
# setup
pip install pytorch-lightning
# 0.6.0 be installed at 2020/02/28

# run your trainer
python ./linear.py

# check your training process
tensorboard --bind_all --port 8888 --logdir ./lightning_logs
```

## NOTE

My pytorch-lightning (`==0.6.0`) will warn...

```
RuntimeWarning: Early stopping conditioned on metric `val_loss` which is not available. Available metrics are: loss,train_loss
```

This may be fixed by [#524](https://github.com/PyTorchLightning/pytorch-lightning/issues/524).
