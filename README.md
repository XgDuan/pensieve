# rnn-mpc

# 文件修改说明

+ mpc.py，由于pensieve代码仅保证了rl_\*代码可以运行，其`test/load_trace`函数返回值与mpc中不一致，因此修改了mpc.py 89行；
+ mpc_future_bandwidth.py, 1) 第二行import fixed_env2 改成fixed_env_future_bandwidth; 2) 11行，26行将MPC_FUTURE_CHUNK_COUNT改成5与mpc保持一致;
+ 修改了大部分固定的文件夹位置参数，使其能够适应不同的实验情况；

# 代码结构
我们在原始pensieve代码基础上，删除不必要的代码，将部分代码中的固定参数值改成了命令行参数（原来的固定文件夹参数等），并增加了新的模块用于训练基于深度学习的带宽预测函数；
+ ./data/* 数据
+ ./data/data_converter.py 数据转换，将下载好的Belgium、hsdpa数据转换成test可以直接使用的格式；使用说明：
  ```
  python data_converter.py raw_data_dir output_dir is_hsdpa_flag
  ```
+ ./data/dataset_* 转换格式后人工划分的训练测试集合
+ deepbdp/* 深度带宽预测包，使用方式见**运行说明**
+ test pensieve中用于测试的文件夹
+ video_server存放视频数据信息，用于仿真测试

# 运行说明

## 数据准备
所有的数据已经处理放在对应的文件夹下，数据转换脚本为`data/data_converter.py`；

## 模型训练
```
cd deepbdp
python train.py --model MODEL_NAME --data_path DATA_PATH
```
比如，利用belgium数据训练基于RNN的模型：
```
python train.py --model rnn --data_path ../data/dataset_belgium
```
此外，train.py其他参数可以通过`python train.py -h`查看，具体参数如下:
```shell
$ python train.py -h
usage: train.py [-h] [--random_seed RANDOM_SEED] [--device {cpu,gpu}]
                [--gpu_id GPU_ID] [--alias ALIAS] [--log_step LOG_STEP]
                [--data_path DATA_PATH] [--result_path RESULT_PATH]
                [--batch_size BATCH_SIZE] [--lr LR]
                [--weight_decay WEIGHT_DECAY] [--max_epoch MAX_EPOCH]
                [--model MODEL] [--hidden_size HIDDEN_SIZE] [--step STEP]
                [--encoder_layer ENCODER_LAYER] [--is_bidir IS_BIDIR]

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
                        random seed used. Note: if you load models from a
                        checkpoint, the random seed would be invalid.
  --device {cpu,gpu}
  --gpu_id GPU_ID
  --alias ALIAS
  --log_step LOG_STEP
  --data_path DATA_PATH
  --result_path RESULT_PATH
  --batch_size BATCH_SIZE
  --lr LR
  --weight_decay WEIGHT_DECAY
  --max_epoch MAX_EPOCH
  --model MODEL         the model to be used, most related parameters is
                        assigned with the model
  --hidden_size HIDDEN_SIZE
  --step STEP
  --encoder_layer ENCODER_LAYER
  --is_bidir IS_BIDIR

```
脚本会自动测试结果，并将训练参数、训练好的模型和训练过程中的错误信息输出到指定文件夹下（默认results文件夹）文件夹下；
## 仿真测试
> 在测试之前，根据pensieve的要求，在test下运行`python get_video_sizes.py`

首先，需要指定测试文件，我们在`test/traces`下存储了belgium，hsdpa和原始的sample文件，将需要的数据集链接到`cooked_traces`即可，比如，使用belgium数据测试如下：

```
cd test
ln -s traces/belgium_traces cooked_traces
python mpc_dbp.py CHECKPOINT
```
其中CHECKPOINT指向上一步骤训练好的模型文件即可；
