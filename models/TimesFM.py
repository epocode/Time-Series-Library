import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
import timesfm


class Model(nn.Module):
    def __init__(self, configs):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()

        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
        # 增加 per_core_batch_size 以提高 GPU 利用率（12GB 显存建议 6-8）
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=configs.seq_len,
                max_horizon=configs.pred_len,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                per_core_batch_size=8,  # 从 4 增加到 8，提高 GPU 利用率
            ),
            torch_compile=True,  # 启用 PyTorch 编译优化
        )

        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        TimesFM forecast method.
        
        重要：TimesFM 期望接收原始数据（未标准化的），然后内部会进行 RevIN 风格的归一化。
        但是数据加载器已经进行了 StandardScaler 标准化，所以我们需要：
        1. 接收标准化后的数据（x_enc）
        2. 直接传递给 TimesFM（TimesFM 会基于标准化数据进行 RevIN 归一化）
        3. TimesFM 输出的是基于标准化数据的反归一化结果（仍然是标准化尺度）
        4. 在评估时，会通过 inverse_transform 转换回原始尺度
        
        注意：TimesFM 的 normalize_inputs=True 会进行 RevIN 归一化（基于每个序列的统计量），
        这与数据加载器的 StandardScaler 标准化（基于训练集的统计量）是不同的。
        """
        B, L, C = x_enc.shape
        device = x_enc.device
        
        # 将多变量时间序列 reshape 为 (B*C, L)，每个通道作为一个独立的时间序列
        x_enc = torch.reshape(x_enc, (B*C, L))

        # TimesFM 的 forecast 方法期望 inputs 是一个列表，每个元素是一维 numpy 数组
        # 注意：虽然数据加载器已经进行了 StandardScaler 标准化，但 TimesFM 内部还会进行
        # RevIN 归一化（基于每个序列的统计量），这是 TimesFM 的设计
        x_enc_np = x_enc.detach().cpu().numpy()
        inputs_list = [x_enc_np[i].copy() for i in range(B*C)]

        # 批量推理
        # TimesFM 内部会：
        # 1. 对每个时间序列进行 RevIN 归一化（基于该序列的均值和方差）
        # 2. 进行预测
        # 3. RevIN 反归一化（输出仍然是标准化后的尺度，因为输入是标准化后的）
        output, _ = self.model.forecast(
            horizon=self.pred_len,
            inputs=inputs_list
        )
        
        # 转换为 tensor 并移动到设备
        output = torch.from_numpy(output).to(device, non_blocking=True)

        # reshape 回 (B, pred_len, C)
        dec_out = torch.reshape(output, (B, output.shape[-1], C))
        
        # 注意：TimesFM 的输出仍然是标准化后的尺度（因为输入是标准化后的）
        # 在评估时会通过 inverse_transform 转换回原始尺度
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'zero_shot_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
