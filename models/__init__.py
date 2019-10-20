import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
    elif model == 'srragan':
        from .SRRaGAN_model import SRRaGANModel as M
    elif model == 'sftgan':
        from .SFTGAN_ACD_model import SFTGAN_ACD_Model as M
    elif model == 'srefgan':
        from .SREFGAN_model import SREFGANModel as M
    elif model == 'sref_rgan':
        from .SREF_RGAN_model import SREF_RGANModel as M
    elif model == 'sra_gan':
        from .SRA_GAN_model import SRA_GANModel as M 
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
