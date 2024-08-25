from . import unet2d
import segmentation_models_pytorch as smp
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def get_base(base_name, exp_dict, n_classes):
    if base_name == "unet2d":
        base = unet2d.UNet(n_channels=1, n_classes=n_classes)

    if base_name == 'pspnet':
        kwargs = {'encoder_name': exp_dict['model']['encoder'],
                  'in_channels': exp_dict['num_channels'],
                  'encoder_weights': None,  # ignore error. it still works.
                  'classes': n_classes}
        if exp_dict['model']['base'] == 'pspnet':
            net_fn = smp.PSPNet

        assert net_fn is not None

        base = smp.PSPNet(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                          encoder_weights=exp_dict['model']['weight'],
                          activation='softmax2d',
                          classes=n_classes)

    if base_name == 'deeplabv3':
        base = smp.DeepLabV3(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                          encoder_weights=exp_dict['model']['weight'],
                          activation='softmax2d',
                          classes=n_classes)

    if base_name == 'deeplabv3plus':
        base = smp.DeepLabV3Plus(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                          encoder_weights=exp_dict['model']['weight'],
                          activation='softmax2d',
                          classes=n_classes)

    if base_name == 'fpn':
        base = smp.FPN(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                        encoder_weights=exp_dict['model']['weight'],
                        activation='softmax2d',
                          classes=n_classes)

    if base_name == 'manet':
        base = smp.MAnet(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                        encoder_weights=exp_dict['model']['weight'],
                        activation='softmax2d', encoder_depth=4,decoder_channels=[256, 128, 64, 32],
                          classes=n_classes, decoder_use_batchnorm='inplace')

    if base_name == 'pan':
        base = smp.PAN(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                        encoder_weights=exp_dict['model']['weight'],
                        activation='softmax2d',encoder_dilation=False,
                          classes=n_classes)

    if base_name == 'unet':
        base = smp.Unet(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                        encoder_weights=exp_dict['model']['weight'],
                        activation='softmax2d',
                          classes=n_classes)

    if base_name == 'unetplus':
        base = smp.UnetPlusPlus(encoder_name=exp_dict['model']['encoder'],
                          in_channels=exp_dict['num_channels'],
                        encoder_weights=exp_dict['model']['weight'],
                        activation='softmax2d',
                          classes=n_classes)

    return base
