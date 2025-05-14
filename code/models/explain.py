import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries

# === Shared Normalization Transform ===
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_default_transform(size=224):
    """Returns a default preprocessing transform for input images."""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        normalize
    ])

def run_lime_interpreter_torch(model, image_np, class_names, device,
                                true_label=None, pred_label=None, filename_prefix=None,
                                save_dir="lime_bagnet", num_features=5, num_samples=1000):
    """
    Runs LIME explainability on a single image.
    image_np should be a (H, W, 3) numpy array, dtype uint8 or float in [0, 1]
    """
    os.makedirs(save_dir, exist_ok=True)

    def predict_fn(imgs):
        imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float() / 255
        imgs = torch.stack([normalize(x) for x in imgs])
        with torch.no_grad():
            outputs = model(imgs.to(device)).softmax(dim=1).cpu().numpy()
        return outputs

    explainer = lime_image.LimeImageExplainer()

    if pred_label is None:
        pred_label = predict_fn(np.expand_dims(image_np, axis=0)).argmax()

    explanation = explainer.explain_instance(
        image_np, predict_fn, labels=[pred_label], num_samples=num_samples, hide_color=0
    )

    temp, mask = explanation.get_image_and_mask(
        label=pred_label, positive_only=True, num_features=num_features, hide_rest=False
    )
    temp = np.array(temp) / 255.0 if temp.max() > 1 else temp

    true_str = class_names[true_label] if true_label is not None else "unknown"
    pred_str = class_names[pred_label]
    name = f"{filename_prefix or 'sample'}_pred_{pred_str}_true_{true_str}.png"
    path = os.path.join(save_dir, name)

    plt.figure()
    plt.imshow(mark_boundaries(temp, mask))
    plt.axis("off")
    plt.title(f"LIME | Pred: {pred_str} | True: {true_str}")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")
    return path

def run_gradcam_interpreter_torch(model, image_np, class_names, device,
                                   true_label=None, pred_label=None, filename_prefix=None,
                                   save_dir="gradcam_bagnet", target_layer="backbone.layer4"):
    """
    Runs Grad-CAM on a single image. image_np should be uint8 (H, W, 3).
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()
    image_t = transforms.ToTensor()(image_np / 255.).unsqueeze(0).to(device)
    image_t = normalize(image_t[0]).unsqueeze(0)

    activations = []
    grads = []

    def hook_activations(module, input, output):
        activations.append(output)

    def hook_grad(module, grad_input, grad_output):
        grads.append(grad_output[0])

    layer = dict([*model.named_modules()])[target_layer]
    h1 = layer.register_forward_hook(hook_activations)
    h2 = layer.register_full_backward_hook(hook_grad)

    output = model(image_t)
    if pred_label is None:
        pred_label = output.argmax(dim=1).item()

    loss = output[0, pred_label]
    model.zero_grad()
    loss.backward()

    act = activations[0][0]
    grad = grads[0][0]
    weights = grad.mean(dim=(1, 2), keepdim=True)
    cam = (weights * act).sum(0).cpu().detach().numpy()
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    heatmap = (plt.cm.jet(cam)[..., :3] * 255).astype("uint8")
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    true_str = class_names[true_label] if true_label is not None else "unknown"
    pred_str = class_names[pred_label]
    name = f"{filename_prefix or 'sample'}_pred_{pred_str}_true_{true_str}.png"
    path = os.path.join(save_dir, name)

    plt.imsave(path, overlay)
    print(f"✅ Saved Grad-CAM to {path}")

    h1.remove()
    h2.remove()
    return path

