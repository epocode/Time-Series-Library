from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Zero_Shot_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Zero_Shot_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        total_batches = len(test_loader)
        print(f'开始测试，共有 {total_batches} 个批次...')
        
        # 批量累积处理优化：累积多个批次后一次性处理，减少 GPU-CPU 转换开销
        batch_accumulator = []
        accumulated_samples = 0
        target_accumulated_samples = 64  # 累积到 64 个样本再处理（可根据显存调整）
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if i % 10 == 0 or i == total_batches - 1:
                    if accumulated_samples > 0:
                        print(f'处理批次 {i+1}/{total_batches}... (累积: {accumulated_samples} 样本)', flush=True)
                    else:
                        print(f'处理批次 {i+1}/{total_batches}...', flush=True)
                
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                # 累积批次
                batch_accumulator.append((batch_x, batch_y, batch_x_mark, batch_y_mark))
                accumulated_samples += batch_x.shape[0]
                
                # 当累积到目标大小或最后一个批次时，批量处理
                if accumulated_samples >= target_accumulated_samples or i == total_batches - 1:
                    # 合并所有累积的批次
                    if len(batch_accumulator) > 0:
                        merged_batch_x = torch.cat([bx for bx, _, _, _ in batch_accumulator], dim=0)
                        merged_batch_y = torch.cat([by for _, by, _, _ in batch_accumulator], dim=0)
                        merged_batch_x_mark = torch.cat([bxm for _, _, bxm, _ in batch_accumulator], dim=0)
                        merged_batch_y_mark = torch.cat([bym for _, _, _, bym in batch_accumulator], dim=0)
                        
                        # decoder input
                        dec_inp = torch.zeros_like(merged_batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([merged_batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                        
                        # 批量推理
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(merged_batch_x, merged_batch_x_mark, dec_inp, merged_batch_y_mark)
                        else:
                            outputs = self.model(merged_batch_x, merged_batch_x_mark, dec_inp, merged_batch_y_mark)
                        
                        # 按原始批次大小分割结果
                        split_sizes = [bx.shape[0] for bx, _, _, _ in batch_accumulator]
                        outputs_list = torch.split(outputs, split_sizes, dim=0)
                        batch_y_list = torch.split(merged_batch_y[:, -self.args.pred_len:, :], split_sizes, dim=0)
                        
                        # 处理每个分割的批次
                        for j, (outputs_split, batch_y_split) in enumerate(zip(outputs_list, batch_y_list)):
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs_np = outputs_split.detach().cpu().numpy()
                            batch_y_np = batch_y_split.detach().cpu().numpy()
                            
                            if test_data.scale and self.args.inverse:
                                shape = batch_y_np.shape
                                if outputs_np.shape[-1] != batch_y_np.shape[-1]:
                                    outputs_np = np.tile(outputs_np, [1, 1, int(batch_y_np.shape[-1] / outputs_np.shape[-1])])
                                outputs_np = test_data.inverse_transform(outputs_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                                batch_y_np = test_data.inverse_transform(batch_y_np.reshape(shape[0] * shape[1], -1)).reshape(shape)

                            outputs_np = outputs_np[:, :, f_dim:]
                            batch_y_np = batch_y_np[:, :, f_dim:]

                            preds.append(outputs_np)
                            trues.append(batch_y_np)
                            
                            # 可视化（每 20 个批次）
                            batch_idx = i - len(batch_accumulator) + j + 1
                            if batch_idx % 20 == 0:
                                bx, _, _, _ = batch_accumulator[j]
                                input_np = bx.detach().cpu().numpy()
                                if test_data.scale and self.args.inverse:
                                    shape = input_np.shape
                                    input_np = test_data.inverse_transform(input_np.reshape(shape[0] * shape[1], -1)).reshape(shape)
                                gt = np.concatenate((input_np[0, :, -1], batch_y_np[0, :, -1]), axis=0)
                                pd = np.concatenate((input_np[0, :, -1], outputs_np[0, :, -1]), axis=0)
                                visual(gt, pd, os.path.join(folder_path, str(batch_idx) + '.pdf'))
                        
                        # 清空累积器
                        batch_accumulator = []
                        accumulated_samples = 0

        print('所有批次处理完成，正在合并结果...', flush=True)
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        print('正在计算评估指标...', flush=True)
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        
        print('正在保存结果...', flush=True)
        f = open("result_zero_shot_forecast_search.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        print('测试完成！', flush=True)
        return
