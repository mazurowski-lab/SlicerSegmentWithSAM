import argparse

def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='sam', help='net type')
    parser.add_argument('-baseline', type=str, default='unet', help='baseline net type')
    parser.add_argument('-seg_net', type=str, default='transunet', help='net type')
    parser.add_argument('-mod', type=str, default='sam_adpt', help='mod type:seg,cls,val_ad')
    parser.add_argument('-exp_name', type=str, default='msa-2d', help='net type')
    parser.add_argument('-type', type=str, default='map', help='condition type:ave,rand,rand_map')
    parser.add_argument('-vis', type=int, default=None, help='visualization')
    parser.add_argument('-reverse', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-pretrain', type=bool, default=False, help='adversary reverse')
    parser.add_argument('-val_freq',type=int,default=100,help='interval between each validation')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-gpu_device', type=int, default=0, help='use which gpu')
    parser.add_argument('-sim_gpu', type=int, default=0, help='split sim to this gpu')
    parser.add_argument('-epoch_ini', type=int, default=1, help='start epoch')
    parser.add_argument('-image_size', type=int, default=1024, help='image_size')
    parser.add_argument('-out_size', type=int, default=256, help='output_size')
    parser.add_argument('-patch_size', type=int, default=2, help='patch_size')
    parser.add_argument('-dim', type=int, default=512, help='dim_size')
    parser.add_argument('-depth', type=int, default=1, help='depth')
    parser.add_argument('-heads', type=int, default=16, help='heads number')
    parser.add_argument('-mlp_dim', type=int, default=1024, help='mlp_dim')
    parser.add_argument('-w', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=8, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-if_warmup', type=bool, default=False, help='if warm up training phase')
    parser.add_argument('-warmup_period', type=int, default=200, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('-uinch', type=int, default=1, help='input channel of unet')
    parser.add_argument('-imp_lr', type=float, default=3e-4, help='implicit learning rate')
    parser.add_argument('-weights', type=str, default = 0, help='the weights file you want to test')
    parser.add_argument('-base_weights', type=str, default = 0, help='the weights baseline')
    parser.add_argument('-sim_weights', type=str, default = 0, help='the weights sim')
    parser.add_argument('-distributed', default='none' ,type=str,help='multi GPU ids to use')
    parser.add_argument('-dataset', default='isic' ,type=str,help='dataset name')
    parser.add_argument('-sam_ckpt', default='checkpoints' , help='sam checkpoint address')
    parser.add_argument('-thd', type=bool, default=False , help='3d or not')
    parser.add_argument('-chunk', type=int, default=96 , help='crop volume depth')
    parser.add_argument('-num_sample', type=int, default=4 , help='sample pos and neg')
    parser.add_argument('-roi_size', type=int, default=96 , help='resolution of roi')
    parser.add_argument('-if_encoder_adapter', type=bool, default=False , help='if add adapter to encoder')
    parser.add_argument('-encoder-adapter-depths', type=list, default=[0,1,10,11] , help='the depth of blocks to add adapter')

    parser.add_argument('-if_mask_decoder_adapter', type=bool, default=False , help='if add adapter to mask decoder')
    parser.add_argument('-decoder_adapt_depth', type=int, default=2, help='the depth of the decoder adapter')
    
    parser.add_argument('-if_encoder_lora_layer', type=bool, default=False , help='if add lora to encoder')
    parser.add_argument('-if_decoder_lora_layer', type=bool, default=False , help='if add lora to decoder')
    parser.add_argument('-encoder_lora_layer', type=list, default=[0,1,10,11] , help='the depth of blocks to add lora')
    parser.add_argument('-if_split_encoder_gpus', type=bool, default=False , help='if split encoder to multiple gpus')
    parser.add_argument('-devices', type=list, default=[0,1] , help='if split encoder to multiple gpus')
    

    parser.add_argument('-if_LST_CNN', type=bool, default=False , help='if add CNN as side net')
    

    parser.add_argument('-evl_chunk', type=int, default=None , help='evaluation chunk')
    parser.add_argument(
    '-data_path',
    type=str,
    default='../data',
    help='The path of segmentation data')
    # '../dataset/RIGA/DiscRegion'
    # '../dataset/ISIC'
    opt = parser.parse_args("")

    return opt
