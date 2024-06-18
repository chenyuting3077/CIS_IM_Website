from mmseg.apis import init_model, inference_model, show_result_pyplot
from thop import profile
import torch

from mmengine.model import revert_sync_batchnorm
config_path = 'configs/segformer/segformer_mit-b2_8xb2-500_IM-224x224.py'
img_path = 'data/IM/images/val/00252209_I_3_leftImg8bit.png'
# build the model from a config file and a checkpoint file
model = init_model(config_path, device='cuda:0')
model = revert_sync_batchnorm(model)
# print(type(model))
# summary(model, (3, 224, 224))
# print(model(torch.randn(1, 3, 224, 224).to("cuda:0")))
# flops, params = profile(model, inputs=torch.randn(3, 224, 224).to("cuda:0"))

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params/1000**2)
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')
# print(model)
# # inference on given image
# result = inference_model(model, img_path)
#
# # display the segmentation result
# vis_image = show_result_pyplot(model, img_path, result)