import cv2
import numpy as np
import torch
from models.optimized_swin2sr import Swin2SR as net
import time

scale = 4
device = "cuda"
model_path = "weights/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
param_key_g = 'params_ema'
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
model.half()
model.eval()
model.to("cuda")

# trace model
# options:

window_size = 8
scale = 4
image = ('test_thumbnail.png')
img_lq = cv2.imread(str(image), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
img_lq = np.transpose(
    img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
)  # HCW-BGR to CHW-RGB
img_lq = (
    torch.from_numpy(img_lq).float().unsqueeze(0).to(device)
)  # CHW-RGB to NCHW-RGB

# inference
with torch.inference_mode():
    # pad input image to be a multiple of window_size
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[
        :, :, : h_old + h_pad, :
    ]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[
        :, :, :, : w_old + w_pad
    ]
    img_lq = img_lq.half()
    # warmup
    for val in range(3):
        fake_input = torch.randn(img_lq.shape).half().to('cuda')
        _ = model(fake_input)
    print("tracing")
    traced_model = torch.jit.trace(model, fake_input)
    traced_model.eval().half()
    model.to('cpu')
    traced_model.to('cuda')

for _ in range(5):
    with torch.inference_mode():
        fake_input = torch.randn(img_lq.shape).half().to('cuda')
        _ = traced_model(fake_input)

with torch.inference_mode():
    for _ in range(5):
        torch.cuda.synchronize()
        start_time = time.time()
        orig_output = traced_model(img_lq)
        torch.cuda.synchronize()
        print(f"traced inference took {time.time() - start_time:.2f} seconds")

    traced_model.to('cpu')
    model.to('cuda')
    for _ in range(5):
        torch.cuda.synchronize()
        start_time = time.time()
        orig_output = model(img_lq)
        torch.cuda.synchronize()
        print(f"untraced inference took {time.time() - start_time:.2f} seconds")

traced_model.save("swin2sr_traced_half.pt")

    






# torch.jit.trace
# torch 2.0.compile() - busted at the moment
# tensorRT

# need - inputs to model
# [b, c, h_pad, w_pad]

# ok. let's try torch 2, why not? 
# approach - throw into docker, hammer, rinse repeat
# benchmark currently locally. 