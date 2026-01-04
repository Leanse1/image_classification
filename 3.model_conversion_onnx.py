import timm
import torch.nn as nn
import torch

MODEL_NAME = "efficientnet_lite0"
NUM_CLASSES = 7   # <-- same as your training

def build_model(num_classes=NUM_CLASSES):
    model = timm.create_model(MODEL_NAME, pretrained=False)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model



model = build_model()
model.load_state_dict(torch.load("efficientnet_lite_best.pth", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 3, 224, 224)


torch.onnx.export(
    model,
    dummy,                              # (1,3,224,224)
    "efficientnet_best.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=18,
    do_constant_folding=True,
    dynamic_axes={
        "input":  {0: "batch"},
        "output": {0: "batch"}
    }
)



print("ONNX exported")

