import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/autonomousvision/gaussian-opacity-fields
def decouple_appearance(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = image.size(1), image.size(2)
    # down sample the image
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode="bilinear", align_corners=True)[0]

    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()
    transformed_image = mapping_image * image

    return transformed_image, mapping_image


class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, H, W):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    H, W = 1200//32, 1600//32
    input_channels = 3 + 64
    output_channels = 3
    input = torch.randn(1, input_channels, H, W).cuda()
    model = AppearanceNetwork(input_channels, output_channels).cuda()
    
    output = model(input)
    print(output.shape)