1.修改项目根目录:在astrodino文件夹中的.env文件里的ASTROCLIP_ROOT

2.修改数据路径:在astrodino文件夹中的config.yaml文件中，修改dataset_path里面root。
【sample_0.1文件夹下有train_dataset和test_dataset】
train:
  batch_size_per_gpu: 72
  dataset_path: QuasarDataset:split=train:root={ASTROCLIP_ROOT}/data/sample_0.1/train_dataset:extinction=False:probs=False

3.检查trainer.py中get_args_parser函数里的命令行参数--config-file路径是否正确(默认:{ASTROCLIP_ROOT}/clip_5_15/astrodino/config.yaml)

4.python astrodino/trainer.py

5.输出的模型保存路径为:f"{ASTROCLIP_ROOT}/outputs/astroclip_image/{run_name}"

6.麻烦老师把模型权重传给我就好了