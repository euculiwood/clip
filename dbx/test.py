import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from dataset_util.PairDataset import PairDataset
from datasets import load_from_disk
from dbx.resnet_z import ResNet18RedshiftPredictor
from torch.utils.data import DataLoader
import torch
import numpy as np
import math

def main(test_data_dir, output_dir,ckpt_path):
    test_data=load_from_disk(test_data_dir)
    test_dataset=PairDataset(test_data)
    test_dataloader=DataLoader(test_dataset,batch_size=256, shuffle=False, num_workers=4)

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model=ResNet18RedshiftPredictor.load_from_checkpoint(ckpt_path)
    model=model.to(device)

    model.eval()
    all_predictions=[]
    all_z=[]
    with torch.no_grad():
        for batch in test_dataloader:
            image = batch['image'].to(device)
            z = batch['z'].to(device)
            prediction = model(image)
            prediction = prediction.squeeze()
            all_predictions.append(prediction.cpu().numpy())
            all_z.append(z.cpu().numpy())

    all_outputs = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_z, axis=0)

    # 计算 delta (相对误差)
    delta = np.abs((all_outputs - all_targets) / (1 + all_targets))

    def get_rms(records):
        return math.sqrt(sum(x ** 2 for x in records) / len(records))

    rms = get_rms(delta)

    # 准确率统计
    total = len(delta)
    count_0_1 = np.sum(delta < 0.1)
    count_0_2 = np.sum(delta < 0.2)
    count_0_3 = np.sum(delta < 0.3)

    ratio_0_1 = count_0_1 / total
    ratio_0_2 = count_0_2 / total
    ratio_0_3 = count_0_3 / total

    log_message = (
        f"RMS: {rms:.4f} | "
        f"Accuracy (<0.1): {ratio_0_1:.4f} | "
        f"Accuracy (<0.2): {ratio_0_2:.4f} | "
        f"Accuracy (<0.3): {ratio_0_3:.4f}"
    )

    print(log_message)

    #将结果保存到文件中,路径为output_dir
    with open(output_dir, 'w') as f:
        f.write(log_message)



if __name__ == '__main__':
    test_data_dir='/hy-tmp/data/sample_0.2/test_dataset'
    output_dir='/hy-tmp/result/dbx_res.txt'
    ckpt_path='/hy-tmp/model_checkpoints/dbx/last.ckpt'
    main(test_data_dir,  output_dir,ckpt_path)
