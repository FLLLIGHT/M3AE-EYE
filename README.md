第一步：下载依赖
pip install -r requirements.txt

第二步：准备数据集
python prepro/prepro_pretraining_data.py
python prepro/prepro_finetuning_data.py
（可选）
如果需要使用自己的数据集，需要添加对应的make_arrow方法，将数据/图像/标签放在dict中存到arrow文件中，
然后在datamodules和dataset里添加对应的加载和划分方法。

第三步：预训练
bash run_scripts/pretrain_m3ae.sh

第四步：下游任务微调
 python main.py with data_root=prepro/data/finetune_arrows num_gpus=1 num_nodes=1 task_finetune_seg_octa500 per_gpu_batchsize=1 clip16 text_roberta image_size=400 clip_resizedcrop load_path=checkpoints/
更多下游任务的配置，请见run_scripts/finetune_m3ae.sh，如需支持其他任务，需要在config里进行添加，
然后再modules里添加对应的任务头，在gadgets/my_metrics.py里修改对应的评价指标等。

由于数据大小原因，处理后的octa500文件和预训练模型不在报告中上传，可以通过网盘链接下载：
https://drive.google.com/drive/folders/1O4wt1q05YJ1wnHPXejrJw86YvNvKjw-N?usp=sharing