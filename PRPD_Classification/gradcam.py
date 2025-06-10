# gradcam.py
import cv2, torch, torch.nn.functional as F, numpy as np
import matplotlib.pyplot as plt

def generate_and_save_gradcam(model, input_tensor, target_layer, class_idx, save_path):
    model.eval()
    activations, gradients = [], []

    # register hooks
    def f_hook(m, i, o): activations.append(o.detach())
    def b_hook(m, gi, go): gradients.append(go[0].detach())
    h_f = target_layer.register_forward_hook(f_hook)
    h_b = target_layer.register_backward_hook(b_hook)

    # forward & backward
    out = model(input_tensor)
    if class_idx is None: class_idx = out.argmax(dim=1).item()
    model.zero_grad()
    one_hot = torch.zeros_like(out); one_hot[0, class_idx]=1
    out.backward(gradient=one_hot)

    # compute CAM
    grads, acts = gradients[0], activations[0]  # (1,C,h,w)
    weights = grads.mean(dim=(2,3), keepdim=True)
    cam = F.relu((weights*acts).sum(dim=1, keepdim=True))
    # cam = F.interpolate(cam, input_tensor.shape[2:], 'bilinear', align_corners=False)[0,0]
    cam = F.interpolate(
        cam,
        size=input_tensor.shape[2:],   # 출력 해상도
        mode='bilinear',               # 보간 모드
        align_corners=False
    )[0,0]

    cam = (cam-cam.min())/(cam.max()-cam.min())

    # prepare images
    img = input_tensor[0].cpu().permute(1,2,0).numpy()
    img = (img*0.229+0.485).clip(0,1)  # ImageNet norm 해제
    heat = cv2.applyColorMap((cam.cpu().numpy()*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)/255.0
    overlay = 0.4*heat + 0.6*img
    overlay = np.clip(overlay, 0.0, 1.0)
    # save plots
    plt.imsave(save_path + '_orig.png', img)
    plt.imsave(save_path + '_cam.png', cam.cpu().numpy(), cmap='jet')
    plt.imsave(save_path + '_overlay.png', overlay)

    h_f.remove(); h_b.remove()
