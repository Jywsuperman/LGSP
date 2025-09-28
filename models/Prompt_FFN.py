import torch
import torch.nn.init as init
import torch.nn as nn

class KeyFFN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, init_type='kaiming'):
        super(KeyFFN, self).__init__()
        # # 还原会原来的维度
        # self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)  # Point-wise Conv
        # self.activation = nn.GELU()
        # self.conv2 = nn.Conv2d(hidden_dim, embed_dim, kernel_size=1)  # Point-wise Conv
        # self.dropout = nn.Dropout(0.2)
        # self.layer_norm = nn.LayerNorm(embed_dim)
        # self.init_type = init_type

        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=1)  # Point-wise Conv
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.init_type = init_type

        # 调用初始化方法
        self._init_weights()

    def _init_weights(self):
        if self.init_type == 'xavier':
            init.xavier_uniform_(self.conv1.weight)
            init.zeros_(self.conv1.bias)
            init.xavier_uniform_(self.conv2.weight)
            init.zeros_(self.conv2.bias)
        elif self.init_type == 'kaiming':
            init.kaiming_uniform_(self.conv1.weight, a=0, mode='fan_in', nonlinearity='relu')
            init.zeros_(self.conv1.bias)
            # init.kaiming_uniform_(self.conv2.weight, a=0, mode='fan_in', nonlinearity='relu')
            # init.zeros_(self.conv2.bias)

    def forward(self, x):
        # # 还原会原来的维度
        # residual = x
        # # Conv2d expects [B, C, H, W], input needs to retain spatial dimensions
        # out = self.conv1(x)  # Expand channel dimension
        # out = self.activation(out)
        # out = self.dropout(out)
        # out = self.conv2(out)  # Reduce back to original channel dimension
        # out = self.dropout(out)
        # # Layer norm across channel dim; permute required for normalization
        # out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        # out = self.layer_norm(out + residual.permute(0, 2, 3, 1))  # Residual + norm
        # out = out.permute(0, 3, 1, 2).contiguous()  # Back to [B, C, H, W]


        # residual = x
        # Conv2d expects [B, C, H, W], input needs to retain spatial dimensions
        out = self.conv1(x)  # Expand channel dimension
        out = self.activation(out)
        out = self.dropout(out)
        # Layer norm across channel dim; permute required for normalization
        out = out.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        out = self.layer_norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()  # Back to [B, C, H, W]
        
        return out
