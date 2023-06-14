import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import pandas as pd


def is_image_file(filename):
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(filename.endswith(ext) for ext in extensions)


def lowlight(test_image_path, reference_image_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_lowlight = Image.open(test_image_path)
    data_reference = Image.open(reference_image_path)

    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.cuda().unsqueeze(0)

    data_reference = (np.asarray(data_reference) / 255.0)
    data_reference = torch.from_numpy(data_reference).float()
    data_reference = data_reference.permute(2, 0, 1)
    data_reference = data_reference.cuda().unsqueeze(0)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load('/content/Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth'))
    start = time.time()
    _, enhanced_image, _ = DCE_net(data_lowlight)

    end_time = (time.time() - start)
    print("End time:", end_time)

    # Calculate metrics
    enhanced_image = enhanced_image.clamp(0, 1)
    psnr = peak_signal_noise_ratio(data_reference[0].cpu().numpy(), enhanced_image[0].cpu().numpy())
    ssim = structural_similarity(data_reference[0].cpu().numpy(), enhanced_image[0].cpu().numpy(), win_size=7,
                                  channel_axis=0)
    mae = np.mean(np.abs(data_reference[0].cpu().numpy() - enhanced_image[0].cpu().numpy()))

    # Print metrics
    print("PSNR:", psnr)
    print("SSIM:", ssim)
    print("MAE:", mae)

    test_image_path = test_image_path.replace('low', 'results')
    result_path = test_image_path
    if not os.path.exists(test_image_path.replace('/' + test_image_path.split("/")[-1], '')):
        os.makedirs(test_image_path.replace('/' + test_image_path.split("/")[-1], ''))

    torchvision.utils.save_image(enhanced_image, result_path)

    return psnr, ssim, mae


if __name__ == '__main__':
    # test_images
    psnr_list = []
    ssim_list = []
    mae_list = []
    with torch.no_grad():
        test_dir = '/content/Zero-DCE/Zero-DCE_code/data/LOLdataset/our485/low'
        reference_dir = '/content/Zero-DCE/Zero-DCE_code/data/LOLdataset/our485/high'

        # Get the list of test image file names
        test_images = os.listdir(test_dir)

        # Iterate over the test image file names
        for test_image_filename in test_images:
            if not is_image_file(test_image_filename):
                continue

            # Construct the corresponding reference image file path
            test_image_path = os.path.join(test_dir, test_image_filename)
            reference_image_filename = test_image_filename
            reference_image_path = os.path.join(reference_dir, reference_image_filename)

            # Process the test and reference images
            psnr, ssim, mae = lowlight(test_image_path, reference_image_path)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            mae_list.append(mae)

    # Save metric results to an Excel file
    data = {'PSNR': psnr_list, 'SSIM': ssim_list, 'MAE': mae_list}
    df = pd.DataFrame(data)
    df.to_excel('metric_results.xlsx', index=False)

    # Calculate average metric values
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    avg_mae = np.mean(mae_list)

    print("Average PSNR:", avg_psnr)
    print("Average SSIM:", avg_ssim)
    print("Average MAE:", avg_mae)
