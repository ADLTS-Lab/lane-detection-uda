import torch
from src.utils.logger import get_logger, success

def convert_to_onnx(model, input_shape, output_path, device, logger=None):
    logger = logger or get_logger()
    model.eval()
    dummy_input = torch.randn(1, 3, *input_shape).to(device)
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      dynamo=False)
    success(logger, "ONNX model saved to %s", output_path)

def convert_to_torchscript(model, input_shape, output_path, device, logger=None):
    logger = logger or get_logger()
    model.eval()
    dummy_input = torch.randn(1, 3, *input_shape).to(device)
    traced = torch.jit.trace(model, dummy_input)
    traced.save(output_path)
    success(logger, "TorchScript model saved to %s", output_path)