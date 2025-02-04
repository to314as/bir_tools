import argparse
from common.args import Args
import torch
import sys
import pathlib
import h5py
import numpy as np
import numpy.fft as nf
import random
import torchvision
from torchvision import transforms, utils
import torch
from torch.nn import functional as F
from pytorch_msssim import ssim
import vd_spiral
import sigpy as sp
import sigpy.plot as pl

sys.argv=['']
def create_arg_parser(parser=None):
    if parser:
        return parser
    parser = argparse.ArgumentParser(description="ML parameters")
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=1, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=40,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', default=True,
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='/mnt/mnt/5TB_slot2/Tobias/Thesis/wrapper',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str, default='/mnt/mnt/5TB_slot2/Tobias/Thesis/wrapper/model.pt',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--logdir', type=str, default='/mnt/mnt/5TB_slot2/Tobias/Thesis/log/wrapper_org',
                        help='Path to an existing checkpoint. Used along with "--resume"')
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--resolution', default=128, type=int, help='Resolution of images')
    parser.add_argument('--device_ids', default=[0,1] , help='GPUS used')
    parser.add_argument('--acceleration', default=4, help='Acceleration factor used in artifical undersampling')
    return parser
args=create_arg_parser().parse_args()
    
def to_tensor(data):
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data).double()

def rsos(data,ax=1):
        return np.sqrt(np.sum(np.square(np.abs(data)),axis=ax))

def to_complex(data):
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]

def make_ift_one(data):
    return nf.ifft(data,axis=0)

def make_ft_one(data):
    return nf.fft(data,axis=0)

def make_ift(data):
    try:
        if len(data.shape)>2:
            return nf.fftshift(nf.ifftn(nf.ifftshift(data),axes=(-2,-1)))
        return nf.fftshift(nf.ifftn(nf.ifftshift(data),axes=(0,-1)))
    except:
        data=data.detach().numpy()
        if len(data.shape)>2:
            return nf.fftshift(nf.ifftn(nf.ifftshift(data),axes=(-2,-1)))
        return nf.fftshift(nf.ifftn(nf.ifftshift(data),axes=(0,-1)))
    return -1

def make_ft(data):
    if len(data.shape)>2:
        return nf.fftshift(nf.fftn(nf.ifftshift(data),axes=(-2,-1)))
    return nf.fftshift(nf.fftn(nf.ifftshift(data),axes=(0,-1)))

def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    if len(data.shape)>2:
        return data[..., w_from:w_to, h_from:h_to]
    else:
        return data[w_from:w_to, h_from:h_to]

def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    if len(data.shape)>2:
        return data[..., w_from:w_to, h_from:h_to, :]
    else:
        return data[w_from:w_to, h_from:h_to, :]

def complex_center_crop_2d(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[w_from:w_to, h_from:h_to, :]

#statistics for normalization
#masking k-space (multiple undersamplings to test)
def random_cartesian_mask(shape=[1,args.resolution,args.resolution],center_fractions=[0.04],accelerations=[4],seed=42):
        rng = np.random.RandomState()
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        rng.seed(seed)
        choice = rng.randint(0, len(accelerations))
        center_fraction = center_fractions[choice]
        acceleration = accelerations[choice]
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
        mask = rng.uniform(size=num_cols) < prob
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        #print(mask.shape)
        return mask

def equi_cartesian_mask(shape=[1,args.resolution,args.resolution],center_fractions=[0.04],accelerations=[8],seed=42):
        rng = np.random.RandomState()
        if len(shape) < 3:
            raise ValueError('Shape should have 3 or more dimensions')
        rng.seed(seed)
        choice = rng.randint(0, len(accelerations))
        center_fraction = center_fractions[choice]
        acceleration = accelerations[choice]
        num_cols = shape[-2]
        num_low_freqs = int(round(num_cols * center_fraction))

        # Create the mask
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad:pad + num_low_freqs] = True

        # Determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (num_low_freqs * acceleration - num_cols)
        offset = rng.randint(0, round(adjusted_accel))

        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = True

        # Reshape the mask
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask

def plain_cartesian_mask(shape=[1,args.resolution,args.resolution],acceleration=args.acceleration):
        mask=np.array([i%acceleration==0 for i in range(shape[-2])])
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = shape[-2]
        mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
        return mask
    
def apply_mask(data,mode="mid",r=4):
    shape = np.array(data.shape)
    if mode=="random":
        c_mask = random_cartesian_mask(shape,accelerations=[r])
    elif mode=="mid":
        c_mask = equi_cartesian_mask(shape,accelerations=[r])
    else:
        c_mask=plain_cartesian_mask(shape,acceleration=r)
    #print(c_mask.shape)
    return data * c_mask, c_mask

def losses(out,tar):
    mse=F.mse_loss(out, tar)
    l1=F.l1_loss(out, tar)
    ssim=ssim( out, tar, data_range=1, size_average=False)
    return mse,l1,ssim

def check_density(traj,d=[],r0=1):
    d_p=[]
    temp=1
    r=r0
    rs=[]
    for i in range(len(traj)):
        #print((traj[i, 0]**2 + traj[i, 1]**2)**0.5)
        #print("r",r)
        if (traj[i, 0]**2 + traj[i, 1]**2)**0.5>r:
            #print(r)
            d.append(temp)
            for i in range(temp-1):
                d_p.append(temp)
            rs.append(r)
            temp=1
            r=np.sqrt(r0+r**2)
            while (traj[i, 0]**2 + traj[i, 1]**2)**0.5>r:
                d.append(temp)
                r=np.sqrt(r0+r**2)
                rs.append(r)
            temp=2
        else:
            temp+=1
    d.append(temp)
    for i in range(temp-1):
        d_p.append(temp)
    #print(d)
    return np.array(d),np.array(rs),np.array(d_p)

def get_dcf(points,dcf,rs):
    for i in range(len(points)):
        for j in range(len(points[i])):
            print(j)
            idx=0
            while (i-args.resolution)**2 + (j-args.resolution)**2>rs[idx]:
                if idx==len(rs)-1:
                    break
                idx+=1
            points[i,j]=points[i,j]*1/dcf[idx]
    return points

def apply_dcf(ksp,dp):
    return ksp/dp

def spiral_undersampling(data,r=8):
    full=data.shape[-2]*data.shape[-1]
    us_factor=r
    N=full//us_factor
    nRounds=100;
    PowCoeff=2; 
    m_size=160;
    p_base=0;
    traj=vd_spiral.makeSpiral(N,nRounds,PowCoeff,p_base,m_size);
    traj=np.swapaxes(traj,0,1)
    img=abs(make_ift(data))
    ksp = sp.nufft(img,traj)
    dcf,rs,d_p=check_density(traj,r0=1)
    img_grid = sp.nufft_adjoint(apply_dcf(ksp,d_p), traj)
    ksp = sp.nufft(img_grid,traj)
    k=make_ift(img_grid)
    #pl.ImagePlot(img_grid,z=0,title='Multi-channel Gridding')
    #pl.ImagePlot(rsos(img_grid,0),title='Multi-channel Gridding')
    return k