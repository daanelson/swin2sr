import argparse
import cv2
import numpy as np
import torch
from cog import BasePredictor, Input, Path

from main_test_swin2sr import define_model, test
from models.optimized_swin2sr import Swin2SR as net


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        # self.device = "cuda:0"

        # args = argparse.Namespace()
        # args.scale = 4
        # args.large_model = False

        # tasks = ["classical_sr", "compressed_sr", "real_sr"]
        # paths = [
        #     "weights/Swin2SR_ClassicalSR_X4_64.pth",
        #     "weights/Swin2SR_CompressedSR_X4_48.pth",
        #     "weights/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
        # ]
        # sizes = [64, 48, 128]

        # self.models = {}
        # for task, path, size in zip(tasks, paths, sizes):
        #     args.training_patch_size = size
        #     args.task, args.model_path = task, path
        #     self.models[task] = define_model(args)
        #     self.models[task].eval()
        #     self.models[task] = self.models[task].to(self.device)
        self.device = "cuda"
        model_path = "weights/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
        self.model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        param_key_g = 'params_ema'
        pretrained_model = torch.load(model_path)
        self.model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
        self.model.eval()
        self.model.to("cuda")
        self.opt_model = torch.compile(self.model, backend='inductor')



    def predict(
        self,
        image: Path = Input(description="Input image"),
        task: str = Input(
            description="Choose a task",
            choices=["classical_sr", "real_sr", "compressed_sr"],
            default="real_sr",
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        model = self.opt_model

        window_size = 8
        scale = 4

        img_lq = cv2.imread(str(image), cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        img_lq = np.transpose(
            img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1)
        )  # HCW-BGR to CHW-RGB
        img_lq = (
            torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)
        )  # CHW-RGB to NCHW-RGB

        # inference
        with torch.no_grad():
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
            import time
            st = time.time()
            print("compiling")
            output = model(img_lq)
            print(f"compile time: {time.time() - st}")

            for p in range(10):
                st = time.time()
                output = model(img_lq)
                print(f"pred time: {time.time() - st}")

            if task == "compressed_sr":
                output = output[0][..., : h_old * scale, : w_old * scale]
            else:
                output = output[..., : h_old * scale, : w_old * scale]

        # save image
        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        if output.ndim == 3:
            output = np.transpose(
                output[[2, 1, 0], :, :], (1, 2, 0)
            )  # CHW-RGB to HCW-BGR
        output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
        output_path = "/tmp/out.png"
        cv2.imwrite(output_path, output)

        return Path(output_path)
    
if __name__ == '__main__':
    p = Predictor()
    p.setup()
    p.predict(image='test_thumbnail.png', task='real_sr')
