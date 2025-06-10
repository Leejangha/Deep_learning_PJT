import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import seaborn as sns
import pandas as pd

def visualize_weight_distribution(net, save_path):
    layers_weights = []
    for m in net.modules():
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
            layers_weights.append(m.weight.data.view(-1).cpu().numpy())

    fig, axs = plt.subplots(len(layers_weights), 1, figsize=(10, 2 * len(layers_weights)))
    for i, layer_weights in enumerate(layers_weights):
        axs[i].hist(layer_weights, bins=50)
        axs[i].set_title(f'Layer {i + 1}')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

def build_model(config, mode, result_path, device):
    if config.model_type == "Resnet":
        from network import ResNet152
        model = ResNet152(output_ch=config.output_ch)
    elif config.model_type == "Efficientnet":
        from network import EfficientNet_b0
        model = EfficientNet_b0(output_ch=config.output_ch)
    elif config.model_type == "Convnext":
        from network import ConvNeXt_Base
        model = ConvNeXt_Base(output_ch=config.output_ch)
    elif config.model_type == "Mobilenet":
        from network import MobileNet_V2
        model = MobileNet_V2(output_ch=config.output_ch)
    elif config.model_type == "Vit":
        from network import ViT_B_16
        model = ViT_B_16(output_ch=config.output_ch)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    if mode == 'train':
        visualize_weight_distribution(model, result_path + 'Weights_Distribution.png')

    if torch.cuda.is_available():
        model.to(device)
    else:
        model.to('cpu')

    return model

def evaluate(acc, SE, PC, F1, result, label):
    acc = accuracy_score(label, result)
    SE = recall_score(label, result, average='weighted')
    PC = precision_score(label, result, average='weighted')
    F1 = f1_score(label, result, average='weighted')
    return acc, SE, PC, F1

def save_confusion_matrix(y_true, y_pred, model_name, result_path):
    label_names = ['normal', 'noise', 'surface', 'corona', 'void']
    conf_matrix = confusion_matrix(y_true, y_pred)

    df_cm = pd.DataFrame(conf_matrix, index=label_names, columns=label_names)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.savefig(result_path + f'{model_name}.png')
    plt.close()
