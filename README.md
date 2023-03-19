# How to Run the Program

## 1. Install Environment

This project can be run on Windows or Ubuntu. The installation environment is very simple. Just execute the following command.

```shell
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

## 2. Data Preparation

Execute one or more of the following commands to download the dataset.

```shell
python data/aishell.py
python data/free_st_chinese_mandarin_corpus.py
python data/thchs_30.py
```

**Note:** The above commands can only be executed under Linux. **If you are under Windows**, you can only first download the dataset locally, then change the `download()` function to the absolute path of the file. After the program is executed, the file text will be automatically decompressed to generate a data list.

```python
# change this line of code
filepath = download(url, md5sum, target_dir)
# to
filepath = "D:\\Download\\data_aishell.tgz"
```

Execute the following command to create a manifest and create a vocabulary.

```shell
python create_manifest.py
```

Output as following:

```shell
-----------  Configuration Arguments -----------
annotation_path: dataset/annotation/
count_threshold: 0
is_change_frame_rate: True
manifest_path: dataset/manifest.train
manifest_prefix: dataset/
max_duration: 20
min_duration: 0
vocab_path: dataset/zh_vocab.json
------------------------------------------------
开始生成数据列表...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 141600/141600 [00:17<00:00, 8321.22it/s]
完成生成数据列表，数据集总长度为178.97小时！
开始生成数据字典...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140184/140184 [00:01<00:00, 89476.12it/s]
数据字典生成完成！
开始抽取1%的数据计算均值和标准值...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 140184/140184 [01:33<00:00, 1507.15it/s]
【特别重要】：均值：-3.146301, 标准值：52.998405, 请根据这两个值修改训练参数！
```

Then modify the `data_mean` and `data_std` in `train.py` according to these two values.

## 3. Train

Just run this command to start training.

```shell
python train.py
```

You can use `python train.py --help` command to view the descriptions and default values of each parameter.

During the training, the program will use VisualDL to record the training results. You can launch VisualDL through the following command.

```shell
visualdl --logdir=log --host 0.0.0.0
```

Then visit `http://localhost:8040` on the browser where you can view the result display.

![Alt text](images/Test%20cer.png)

## 4. Evaluate and Infer

Run this command to evaluate the model.

```shell
python eval.py --model_path=models/step_final/
```

You can use `python eval.py --help` command to view the descriptions and default values of each parameter.

Run this command to do speech recognition for audio file.

```shell
python infer.py --audio_path=./dataset/test.wav
```

You can use `python infer.py --help` command to view the descriptions and default values of each parameter.
