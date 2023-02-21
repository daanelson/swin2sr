#!/bin/bash

curl -o weights/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth -L https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth
curl -o weights/Swin2SR_CompressedSR_X4_48.pth -L https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_CompressedSR_X4_48.pth
curl -o weights/Swin2SR_ClassicalSR_X4_64.pth -L https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth