import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = F.pad(input_data, [pad, pad, pad, pad], 'constant', 0)
    col = torch.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = torch.permute(col, (0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w, -1)
    return col

def im2col_from_conv(input_data, conv):
    return im2col(input_data, conv.kernel_size[0], conv.kernel_size[1], conv.stride[0], conv.padding[0])


def get_params(model, recurse=False):
    """Returns dictionary of paramters

    Arguments:
        model {torch.nn.Module} -- Network to extract the parameters from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params


def get_buffers(model, recurse=False):
    """Returns dictionary of buffers

    Arguments:
        model {torch.nn.Module} -- Network to extract the buffers from

    Keyword Arguments:
        recurse {bool} -- Whether to recurse through children modules

    Returns:
        Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                   associated parameter arrays
    """
    buffers = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_buffers(recurse=recurse)}
    return buffers

def shrink_cov(cov):
    diag_mean = torch.mean(torch.diagonal(cov))
    off_diag = cov.clone()
    off_diag.fill_diagonal_(0.0)
    mask = off_diag != 0.0
    off_diag_mean = (off_diag*mask).sum() / mask.sum()
    iden = torch.eye(cov.shape[0], device=cov.device)
    alpha1 = 1
    alpha2  = 1
    cov_ = cov + (alpha1*diag_mean*iden) + (alpha2*off_diag_mean*(1-iden))
    return cov_
def sample(mean, cov, size, shrink=False):
    vec = torch.randn(size, mean.shape[-1], device=mean.device)
    if shrink:
        cov = shrink_cov(cov)
    sqrt_cov = torch.linalg.cholesky(cov)
    vec = vec @ sqrt_cov.t()
    vec = vec + mean
    return vec


class AddNoise(nn.Module):
    def __init__(self, scale=0.1):
        super(AddNoise, self).__init__()
        self.scale = scale

    def forward(self, x):
        noise = torch.randn_like(x) * self.scale
        return x + noise
    

# import torch
# import torch.nn as nn

class AdaptiveFrequencyMask(nn.Module):
    def __init__(self, input_size, num_points, args):
        super().__init__()
        self.input_size = input_size  # 输入大小 (h, w)
        self.num_points = num_points  # 控制点数量
        self.args = args

        # 初始化控制点位置为均匀分布
        h, w = input_size
        y_coords = torch.linspace(0, h - 1, steps=int(num_points ** 0.5))
        x_coords = torch.linspace(0, w - 1, steps=int(num_points ** 0.5))
        grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        control_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [num_points, 2]
        
        # self.control_points = nn.Parameter(control_points)  # 形状为 [num_points, 2]
        self.control_points = control_points

        # 掩码的权重
        self.weights = nn.Parameter(torch.ones(num_points))

        # print(self.control_points)
        # print(self.weights)
        # sys.exit()

    def forward(self, input):
        return self.get_deformable_mask(input), self.control_points, self.weights

    def get_deformable_mask(self, input):
        h, w = self.input_size
        device = input.device

        # 创建网格以计算掩码
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        grid = torch.stack([x, y], dim=-1)  # [h, w, 2]

        # 计算每个像素到控制点的距离
        control_points = self.control_points.to(device)  # 确保控制点在同一设备
        # 确保 grid 和 control_points 是浮点类型
        grid = grid.float()  # 转为浮点类型
        control_points = control_points.float()  # 转为浮点类型
        distances = torch.cdist(grid.view(-1, 2), control_points)  # [h*w, num_points]
        distances = distances.view(h, w, -1)  # [h, w, num_points]

        # 生成动态掩码
        beta = 4.0  # 控制掩码锐度的超参数
        masks = torch.sigmoid(-beta * distances)  # [h, w, num_points]

        # 权重归一化
        weights_normalized = torch.softmax(self.weights, dim=0)  # [num_points]
        weighted_masks = weights_normalized * masks  # [h, w, num_points]

        # 合成最终掩码
        final_mask = weighted_masks.sum(dim=-1)  # [h, w]
        return final_mask
