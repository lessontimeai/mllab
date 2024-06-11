import torch
from PIL import Image
import numpy as np

image_path = r"C:\Users\halid\OneDrive\Desktop\Sample-of-24-bit-RGB-of-graphical-image-with-512512-pixels-N-2-i-j-i1-j-1-k.png"
image = Image.open(image_path).convert('RGB')

model_version="radio_v2.1" # for RADIO
#model_version="e-radio_v2" # for E-RADIO
model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_version, progress=True, skip_validation=True)
model.eval()
#x = torch.rand(1, 3, 512, 512)
x = torch.tensor(np.array(image)).permute([2, 0, 1]).unsqueeze(0).half()/255

if "e-radio" in model_version:
    model.model.set_optimal_window_size(x.shape[2:]) #where it expects a tuple of (height, width) of the input image.

# RADIO expects the input to have values between [0, 1]. It will automatically normalize them to have mean 0 std 1.
summary, spatial_features = model(x)

# RADIO also supports running in mixed precision:
with torch.autocast('cpu', dtype=torch.bfloat16):
    summary, spatial_features = model(x)

# If you'd rather pre-normalize the inputs, then you can do this:
conditioner = model.make_preprocessor_external()

# Now, the model won't change the inputs, and it's up to the user to call `cond_x = conditioner(x)` before
# calling `model(cond_x)`. You most likely would do this if you want to move the conditioning into your
# existing data processing pipeline.
with torch.autocast('cpu', dtype=torch.bfloat16):
    cond_x = conditioner(x)
    summary, spatial_features = model(cond_x)