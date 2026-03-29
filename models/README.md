
# Pre-trained Models
## Files

After running the pipeline (especially `--mode all` or `--mode evaluate`), you will find:

- `best_source_model.pth` – Model trained only on the source domain (supervised).
- `best_adapted_model.pth` – Model after unsupervised domain adaptation (if adaptation was performed).
- `final_lane_model.pth` – The model selected as final (either source-only or adapted, whichever achieves higher mIoU on the target test set).
- `final_lane_model.onnx` – Final model exported to ONNX format for inference with ONNX Runtime.
- `final_lane_model.pt` – Final model exported to TorchScript for deployment in production environments.

## Model Architecture

The model is a U‑Net with a ResNet34 encoder pre‑trained on ImageNet. It outputs two classes: background and lane.

## Usage

### Loading a model for inference

```python
import torch
from src.models.unet_resnet import UNetWithResNetEncoder

model = UNetWithResNetEncoder(n_classes=2)
model.load_state_dict(torch.load('models/pretrained/final_lane_model.pth', map_location='cpu'))
model.eval()
```
### Running inference with ONNX
```bash
python -c "import onnxruntime as ort; session = ort.InferenceSession('models/pretrained/final_lane_model.onnx')"
```
