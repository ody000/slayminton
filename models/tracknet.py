import os
import cv2
import numpy as np
import torch
import torch.nn as nn

class Conv(nn.Module):
    """Convolutional block with BatchNorm and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, activation=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU() if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class TrackNet(nn.Module):
    def __init__(self, out_channels=3):
        super(TrackNet, self).__init__()
        
        # Encoder: 9-channel input (3 stacked RGB frames)
        self.conv2d_1 = Conv(9, 64)
        self.conv2d_2 = Conv(64, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2d_3 = Conv(64, 128)
        self.conv2d_4 = Conv(128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2d_5 = Conv(128, 256)
        self.conv2d_6 = Conv(256, 256)
        self.conv2d_7 = Conv(256, 256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2d_8 = Conv(256, 512)
        self.conv2d_9 = Conv(512, 512)
        self.conv2d_10 = Conv(512, 512)
        
        # Decoder with skip connections
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_11 = Conv(768, 256)  # 512 + 256 (skip)
        self.conv2d_12 = Conv(256, 256)
        self.conv2d_13 = Conv(256, 256)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_14 = Conv(384, 128)  # 256 + 128 (skip)
        self.conv2d_15 = Conv(128, 128)
        
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2d_16 = Conv(192, 64)  # 128 + 64 (skip)
        self.conv2d_17 = Conv(64, 64)
        
        self.conv2d_18 = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.conv2d_2(self.conv2d_1(x))
        x_pool1 = self.maxpool1(x1)
        
        x2 = self.conv2d_4(self.conv2d_3(x_pool1))
        x_pool2 = self.maxpool2(x2)
        
        x3 = self.conv2d_7(self.conv2d_6(self.conv2d_5(x_pool2)))
        x_pool3 = self.maxpool3(x3)
        
        x4 = self.conv2d_10(self.conv2d_9(self.conv2d_8(x_pool3)))
        
        # Decoder with skip connections
        up1 = self.upsample1(x4)
        concat1 = torch.cat([up1, x3], dim=1)
        d1 = self.conv2d_13(self.conv2d_12(self.conv2d_11(concat1)))
        
        up2 = self.upsample2(d1)
        concat2 = torch.cat([up2, x2], dim=1)
        d2 = self.conv2d_15(self.conv2d_14(concat2))
        
        up3 = self.upsample3(d2)
        concat3 = torch.cat([up3, x1], dim=1)
        d3 = self.conv2d_17(self.conv2d_16(concat3))
        
        out = self.sigmoid(self.conv2d_18(d3))
        return out


class TrackNetTracker:
    """Lightweight wrapper around TrackNet model to provide a `detect(frame, timestamp)` API.

    Assumptions/behavior:
    - The TrackNet model expects a 9-channel input (3 consecutive RGB frames). For
      simplicity we replicate the single RGB frame three times to create a 9-channel
      tensor when only one frame is available.
    - The model output is treated as a heatmap (use channel 0 or the first channel).
      We locate the argmax and return a fixed-size box centered at that location.
    - Returned detection dict follows existing code expectations: keys `shuttle` and `player`.
      `shuttle` is a tuple (timestamp, x, y, w, h) with x,y as top-left pixel coords.
    """

    def __init__(self, weights_path: str = None, device: str = "cpu", box_size: int = 16):
        self.device = torch.device(device if isinstance(device, str) else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Determine output channels from checkpoint or use default (1 for heatmap)
        out_ch = 1
        state = None
        if weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location="cpu")
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                # Look for final conv layer weight shape to infer output channels
                for key in reversed(list(sd.keys())):
                    if isinstance(sd[key], torch.Tensor) and sd[key].ndim == 4:
                        out_ch = int(sd[key].shape[0])
                        break
                print(f"[TRACKNET] Loaded checkpoint suggests out_channels={out_ch}")
            except Exception as e:
                print(f"[TRACKNET] Warning: failed to inspect weights ({e}), using default out_channels=1")

        self.model = TrackNet(out_channels=out_ch)
        if state is not None:
            try:
                sd = state.get("state_dict", state) if isinstance(state, dict) else state
                self.model.load_state_dict(sd, strict=False)
                print(f"[TRACKNET] Loaded weights from {weights_path}")
            except Exception as e:
                print(f"[TRACKNET] Warning: failed to load weights ({e}), proceeding with random init")
        else:
            if weights_path:
                print(f"[TRACKNET] Warning: weights not found at {weights_path}; using random init")
        
        self.model.to(self.device).eval()
        self.box_size = int(box_size)
        # Standard TrackNet expected resolution
        self.expected_size = (288, 512)  # (height, width)

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        # frame can be either a single HxWx3 RGB array or an iterable of 3 frames.
        if isinstance(frame, (list, tuple)):
            frames = frame
        else:
            frames = [frame, frame, frame]

        arrays = []
        for f in frames:
            img = f.astype(np.float32) / 255.0
            chw = np.transpose(img, (2, 0, 1))
            arrays.append(chw)
        # concatenate along channel dim => 9 x H x W
        stacked = np.concatenate(arrays, axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        return tensor

    def _postprocess_heatmap(self, heatmap: np.ndarray, frame_w: int, frame_h: int):
        # heatmap assumed same HxW as frame (or close). Resize if needed.
        if heatmap.shape[0] != frame_h or heatmap.shape[1] != frame_w:
            heatmap = cv2.resize(heatmap, (frame_w, frame_h))
        # find max
        minv, maxv, minloc, maxloc = cv2.minMaxLoc(heatmap.astype(np.float32))
        cx, cy = int(maxloc[0]), int(maxloc[1])
        half = self.box_size // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        w = min(self.box_size, frame_w - x0)
        h = min(self.box_size, frame_h - y0)
        return x0, y0, w, h, float(maxv)

    def detect(self, frame: np.ndarray, timestamp: float = 0.0) -> dict:
        """Run TrackNet on one RGB frame and return detections dict.

        Returns: {"shuttle": (timestamp, x, y, w, h)} or {} if nothing found.
        """
        # support a list/tuple of 3 frames or a single frame
        if isinstance(frame, (list, tuple)):
            frames = frame
            first = frames[0]
        else:
            frames = [frame, frame, frame]
            first = frame

        # Resize input frames to expected model resolution to match BatchNorm spatial hack
        h, w = first.shape[:2]
        eh, ew = self.expected_size
        if (h, w) != (eh, ew):
            resized = [cv2.resize(f, (ew, eh)) for f in frames]
        else:
            resized = frames

        inp = self._preprocess(resized)
        with torch.no_grad():
            out = self.model(inp)
            # out: (1, C, H, W)
            out_np = out.squeeze(0).cpu().numpy()
        # choose first channel if multiple
        if out_np.ndim == 3:
            heat = out_np[0]
        else:
            heat = out_np
        x0, y0, bw, bh, conf = self._postprocess_heatmap(heat, w, h)
        # If confidence very low, return empty
        if conf < 1e-3:
            return {}
        return {"shuttle": (float(timestamp), float(x0), float(y0), float(bw), float(bh))}