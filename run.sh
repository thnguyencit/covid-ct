# Please run these commands to reproduce 20 configurations
python CovidSeg/trainval.py --encoder timm-efficientnet-b0 --base unet
python CovidSeg/trainval.py --encoder timm-efficientnet-b0 --base unetplus
python CovidSeg/trainval.py --encoder timm-efficientnet-b0 --base unet2d
python CovidSeg/trainval.py --encoder timm-efficientnet-b0 --base fpn

python3 CovidSeg/trainval.py --encoder resnet34 --base unet
python CovidSeg/trainval.py --encoder resnet34 --base unetplus
python CovidSeg/trainval.py --encoder resnet34 --base unet2d
python CovidSeg/trainval.py --encoder resnet34 --base fpn

python CovidSeg/trainval.py --encoder timm-res2net50_26w_4s --base unet
python CovidSeg/trainval.py --encoder timm-res2net50_26w_4s --base unetplus
python CovidSeg/trainval.py --encoder timm-res2net50_26w_4s --base unet2d
python CovidSeg/trainval.py --encoder timm-res2net50_26w_4s --base fpn

python CovidSeg/trainval.py --encoder timm-resnest26d --base unet
python CovidSeg/trainval.py --encoder timm-resnest26d --base unetplus
python CovidSeg/trainval.py --encoder timm-resnest26d --base unet2d
python CovidSeg/trainval.py --encoder timm-resnest26d --base fpn

python CovidSeg/trainval.py --encoder se_resnext50_32x4d --base unet
python CovidSeg/trainval.py --encoder se_resnext50_32x4d --base unetplus
python CovidSeg/trainval.py --encoder se_resnext50_32x4d --base unet2d
python CovidSeg/trainval.py --encoder se_resnext50_32x4d --base fpn