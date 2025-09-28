
from timm.models.registry import register_model

# 这样就保证了只使用了vision_transformer中的_create_vision_transformer函数
# 而其他的模型没有被register，也可以使用pretrain哈哈
from models.vision_transformer_My import _create_vision_transformer
from models.vision_transformer_My import VisionTransformer

import torch.nn as nn
__all__ = [
    'vit_tiny_patch16_224', 'vit_small_patch16_224',  
    'vit_base_patch16_224_in21k', 'vit_base_patch16_224_sam', 'vit_base_patch16_224_dino', 'vit_base_patch16_224_miil_in21k',

    'vit_base_patch16_224', 'vit_base_patch16_224_miil', 
    'vit_base_patch16_clip_224','vit_base_patch16_clip_quickgelu_224'
]


@register_model
def vit_tiny_patch16_224(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_small_patch16_224(pretrained=False, **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

# @register_model
# def vit_base_patch16_224(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
#     return model

@register_model
def vit_base_patch16_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vit_base_patch16_224_in21k(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has valid 21k classifier head and no representation (pre-logits) layer
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_in21k', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_sam(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) w/ SAM pretrained weights. Paper: https://arxiv.org/abs/2106.01548
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_sam', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_dino(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) /w DINO pretrained weights (no head) - https://arxiv.org/abs/2104.14294
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_dino', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def vit_base_patch16_224_miil_in21k(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_miil_in21k', pretrained=pretrained, **model_kwargs)
    return model


# @register_model
# def vit_base_patch16_224_miil(pretrained=False, **kwargs):
#     """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
#     Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
#     """
#     model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False, **kwargs)
#     model = _create_vision_transformer('vit_base_patch16_224_miil', pretrained=pretrained, **model_kwargs)
#     return model
@register_model
def vit_base_patch16_224_miil(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=False)
    model = _create_vision_transformer(
        'vit_base_patch16_224_miil', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

# 上面我引入的都是被弃用了的，接下来我尝试一下最新文件中的

@register_model
def vit_base_patch16_clip_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-B/16 CLIP image tower
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True, norm_layer=nn.LayerNorm)
    model = _create_vision_transformer(
        'vit_base_patch16_clip_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

@register_model
def vit_base_patch16_clip_quickgelu_224(pretrained: bool = False, **kwargs) -> VisionTransformer:
    """ ViT-B/16 CLIP image tower w/ QuickGELU act
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, pre_norm=True,
        norm_layer=nn.LayerNorm, act_layer='quick_gelu')
    model = _create_vision_transformer(
        'vit_base_patch16_clip_quickgelu_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model