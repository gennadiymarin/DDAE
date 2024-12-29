import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetWithResnetEncoder(nn.Module):
    def __init__(self, encoder, out_channels=1):
        """
        Initialize the UNet-like model using a given ResNet-based encoder.

        Args:
            encoder (nn.Module): A ResNet (e.g., ResNet50) where .fc is replaced by Identity().
            out_channels (int): Number of output channels, 1 for depth estimation.
        """
        super(UNetWithResnetEncoder, self).__init__()

        self.encoder = encoder

        self.enc_conv1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )
        self.enc_pool1 = self.encoder.maxpool
        self.enc_layer1 = self.encoder.layer1
        self.enc_layer2 = self.encoder.layer2
        self.enc_layer3 = self.encoder.layer3
        self.enc_layer4 = self.encoder.layer4


        self.up4 = self._upsample_block(2048, 1024)
        self.dec4 = self._conv_block(2048, 1024)

        self.up3 = self._upsample_block(1024, 512)
        self.dec3 = self._conv_block(1024, 512)

        self.up2 = self._upsample_block(512, 256)
        self.dec2 = self._conv_block(512, 256)

        self.up1 = self._upsample_block(256, 64)
        self.dec1 = self._conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        """ A simple 2x conv block. """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def _upsample_block(self, in_channels, out_channels):
        """ Transposed convolution or bilinear upsampling + conv to reduce channels. """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):

        x1 = self.enc_conv1(x)
        x1p = self.enc_pool1(x1)

        x2 = self.enc_layer1(x1p)
        x3 = self.enc_layer2(x2)
        x4 = self.enc_layer3(x3)
        x5 = self.enc_layer4(x4)


        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)


        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)


        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)


        out = self.final_conv(d1)


        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)

        return out

class DiffusionEncoder(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, imgs, timestep, class_labels=None, up_last=-1):
        params = 0

        if self.unet.config.center_input_sample:
            imgs = 2 * imgs - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=imgs.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(imgs.device)

        timesteps = timesteps * torch.ones(imgs.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.unet.time_proj(timesteps)


        t_emb = t_emb.to(dtype=self.unet.dtype)
        emb = self.unet.time_embedding(t_emb)

        total = get_parameters(self.unet.time_embedding)[0]
        params += total
        # print(f'time_embedding {total}')

        if self.unet.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.unet.config.class_embed_type == "timestep":
                class_labels = self.unet.time_proj(class_labels)

            class_emb = self.unet.class_embedding(class_labels).to(dtype=self.unet.dtype)
            emb = emb + class_emb

            total = get_parameters(self.unet.class_embedding)[0]
            params += total
            # print(f'time_embedding {total}')
        elif self.unet.class_embedding is None and class_labels is not None:
            raise ValueError("class_embedding needs to be initialized in order to use class conditioning")

        # 2. pre-process
        skip_sample = imgs
        imgs = self.unet.conv_in(imgs)

        total = get_parameters(self.unet.conv_in)[0]
        params += total



        # 3. down
        down_block_res_samples = (imgs,)
        for downsample_block in self.unet.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                imgs, res_samples, skip_sample = downsample_block(
                    hidden_states=imgs, temb=emb, skip_sample=skip_sample
                )
            else:
                imgs, res_samples = downsample_block(hidden_states=imgs, temb=emb)

            down_block_res_samples += res_samples

            total = get_parameters(downsample_block)[0]
            params += total


        # 4. mid
        imgs = self.unet.mid_block(imgs, emb)

        total = get_parameters(self.unet.mid_block)[0]
        params += total
        # print(f'mid_block {total}')

        # 5. up
        skip_sample = None
        for i, upsample_block in enumerate(self.unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                imgs, skip_sample = upsample_block(imgs, res_samples, emb, skip_sample)
            else:
                imgs = upsample_block(imgs, res_samples, emb)

            total = get_parameters(upsample_block)[0]
            params += total
            # print(f'upsample_block {total}')

            if up_last == i:
                # print(f'params used = {readable_number(params)}')
                return imgs.mean(dim=[2, 3])


        # 6. post-process
        imgs = self.unet.conv_norm_out(imgs)
        imgs = self.unet.conv_act(imgs)
        imgs = self.unet.conv_out(imgs)

        if skip_sample is not None:
            imgs += skip_sample

        if self.unet.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((imgs.shape[0], *([1] * len(imgs.shape[1:]))))
            imgs = imgs / timesteps

        return imgs

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1)
        )

        self.skip_conv = nn.Conv2d(skip_channels, out_channels, 1)

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = self.upsample(x)
        skip = self.skip_conv(skip)



        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DiffusionUNet(nn.Module):
    def __init__(self, diffusion_backbone, out_channels=1):
        super().__init__()
        self.backbone = diffusion_backbone


        self.channels = {
            'down1': 128,
            'down2': 256,
            'down3': 512,
            'mid': 512
        }

        print("Channel sizes:", self.channels)


        self.up3 = UpBlock(
            in_channels=self.channels['mid'],      # 512
            out_channels=self.channels['down3'],   # 512
            skip_channels=self.channels['down2']   # 256
        )
        self.up2 = UpBlock(
            in_channels=self.channels['down3'],    # 512
            out_channels=self.channels['down2'],   # 256
            skip_channels=self.channels['down2']   # 256
        )
        self.up1 = UpBlock(
            in_channels=self.channels['down2'],    # 256
            out_channels=self.channels['down1'],   # 128
            skip_channels=self.channels['down1']   # 128
        )


        self.final = nn.Sequential(
            nn.Conv2d(self.channels['down1'], 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 1)
        )

    @torch.amp.autocast('cuda')
    def forward(self, x):
        torch.cuda.empty_cache()

        batch_size = x.shape[0]
        t = torch.zeros(batch_size, dtype=torch.long, device=x.device)

        with torch.no_grad():
            t_emb = self.backbone.unet.time_proj(t)
            t_emb = t_emb.to(dtype=self.backbone.unet.dtype)
            emb = self.backbone.unet.time_embedding(t_emb)

            if self.backbone.unet.config.center_input_sample:
                x = 2 * x - 1.0

            x = self.backbone.unet.conv_in(x)

            skip_connections = []

            for block in self.backbone.unet.down_blocks:
                if hasattr(block, "skip_conv"):
                    x, res_samples, skip_sample = block(
                        hidden_states=x,
                        temb=emb,
                        skip_sample=None
                    )
                else:
                    x, res_samples = block(hidden_states=x, temb=emb)

                skip_connections.append(x.detach())


            x = self.backbone.unet.mid_block(x, emb)


        skip_connections = skip_connections[:-2]
        skip_connections = skip_connections[::-1]

        for i, (up_block, skip) in enumerate(zip([self.up3, self.up2, self.up1], skip_connections)):
            x = up_block(x, skip)

        depth = self.final(x)
        depth = F.interpolate(depth, size=(256, 256), mode='bilinear', align_corners=False)

        return depth