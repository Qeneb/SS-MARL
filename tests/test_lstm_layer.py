import torch
import torch.nn as nn
from ssmarl.algorithms.utils.rnn import LSTMLayer


def test_lstm_layer():
    # 定义输入参数
    batch_size = 4
    seq_len = 10
    input_dim = 8
    hidden_dim = 16
    recurrent_N = 2
    use_orthogonal = True

    # 创建 LSTMLayer 实例
    lstm_layer = LSTMLayer(input_dim, hidden_dim, recurrent_N, use_orthogonal)

    # 创建输入数据
    x = torch.randn(seq_len * batch_size, input_dim)
    hxs = (
        torch.zeros(batch_size, recurrent_N, hidden_dim),
        torch.zeros(batch_size, recurrent_N, hidden_dim)
    )
    masks = torch.ones(seq_len * batch_size, 1)

    # 前向传播
    output, (hxs_out, cxs_out) = lstm_layer(x, hxs, masks)

    # 验证输出形状
    assert output.shape == (seq_len * batch_size, hidden_dim), "Output shape mismatch"
    assert hxs_out.shape == (batch_size, recurrent_N, hidden_dim), "Hidden state shape mismatch"
    assert cxs_out.shape == (batch_size, recurrent_N, hidden_dim), "Cell state shape mismatch"

    print("LSTMLayer test passed!")


if __name__ == "__main__":
    test_lstm_layer()
