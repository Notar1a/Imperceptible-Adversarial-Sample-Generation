#替代模型默认改为了STARNET
import os
import string
import argparse
import numpy as np
import torch
import cv2
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from datetime import datetime
from STR_modules.prediction import CTCLabelConverter, AttnLabelConverter
from STR_modules.model import Model
from skimage.metrics import structural_similarity as ssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, imgH, imgW):
        self.folder_path = folder_path
        self.imgH = imgH
        self.imgW = imgW
        self.image_paths = [os.path.join(folder_path, img) for img in os.listdir(folder_path) if
                            img.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transforms.Compose([
            transforms.Resize((self.imgH, self.imgW)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, img_path


def load_pretrained_model(model, saved_model_path):
    state_dict = torch.load(saved_model_path, map_location=device)
    model_state_dict = model.state_dict()

    filtered_state_dict = {k: v for k, v in state_dict.items() if
                           k in model_state_dict and model_state_dict[k].shape == v.shape}

    model_state_dict.update(filtered_state_dict)
    model.load_state_dict(model_state_dict)

def train_substitute_model(substitute_model, original_model, dataloader, converter, opt, num_epochs=5):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(substitute_model.parameters(), lr=0.001)

    load_pretrained_model(substitute_model, 'STR_modules/downloads_models/STARNet-TPS-ResNet-BiLSTM-CTC-sensitive.pth')  # 加载你的模型

    for epoch in range(num_epochs):
        substitute_model.train()
        for imgs, _ in dataloader:
            imgs = imgs.to(device)

            if 'CTC' in opt.Prediction:
                length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
                with torch.no_grad():
                    preds = original_model(imgs, text_for_pred).log_softmax(2)
                    preds_size = torch.IntTensor([preds.size(1)])
                    _, preds_index = preds.max(2)
                    targets = preds_index.squeeze(1).to(device)

                outputs = substitute_model(imgs, text_for_pred).log_softmax(2)
                loss = criterion(outputs.permute(1, 2, 0), targets.permute(1, 0))

            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
                with torch.no_grad():
                    preds = original_model(imgs, text_for_pred, is_train=False)
                    _, preds_index = preds.max(2)
                    targets = preds_index.squeeze(1).to(device)

                outputs = substitute_model(imgs, text_for_pred, is_train=False)
                loss = criterion(outputs.permute(1, 2, 0), targets.permute(1, 0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

def calculate_psnr(img1, img2):
    mse = calculate_mse(img1, img2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    return ssim(img1, img2, multichannel=True, win_size=3, data_range=255, channel_axis=2)

def calculate_mse(img1, img2):
    """计算均方误差（MSE）"""
    return np.mean((img1 - img2) ** 2)

def adaptive_local_search_attack(image, model, converter, opt, num_iterations=10, alpha=0.03, decay_factor=0.99,
                                 min_alpha=0.001, max_alpha=0.05, use_perceptual_loss=True):
    image.requires_grad = True
    original_image = image.clone().detach()
    initial_alpha = alpha
    optimizer = torch.optim.Adam([image], lr=alpha)

    for i in range(num_iterations):
        image.requires_grad = True
        model.train()

        if 'CTC' in opt.Prediction:
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(image, text_for_pred).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(image, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_str = preds_str[0][:preds_str[0].find('[s]')]

        loss = -torch.mean(preds)

        if use_perceptual_loss:
            perceptual_loss = F.mse_loss(image, original_image)
            loss += perceptual_loss

        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            perturbation = alpha * torch.sign(image.grad)
            perturbation += torch.randn_like(image) * alpha * 0.05
            image = image + perturbation
            image = torch.clamp(image, original_image - alpha, original_image + alpha)
            image = torch.clamp(image, 0, 1)

        image = image.detach()

        # 检查图像质量并根据结果调整alpha值
        downsampled_img = transforms.Resize((32, 100))(image)
        downsampled_img_np = downsampled_img.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
        downsampled_img_np = downsampled_img_np.astype(np.uint8)

        original_img_resized = transforms.Resize((32, 100))(original_image)
        original_img_np = original_img_resized.squeeze().cpu().numpy().transpose(1, 2, 0) * 255
        original_img_np = original_img_np.astype(np.uint8)

        # 转换为灰度图像
        downsampled_img_gray = cv2.cvtColor(downsampled_img_np, cv2.COLOR_RGB2GRAY)
        original_img_gray = cv2.cvtColor(original_img_np, cv2.COLOR_RGB2GRAY)

        psnr_value = calculate_psnr(downsampled_img_gray, original_img_gray)
        ssim_value = calculate_ssim(downsampled_img_np, original_img_np)
        mse_value = calculate_mse(downsampled_img_gray, original_img_gray)

        if psnr_value >= 40 and ssim_value >= 0.9 and mse_value < 100:
            # 如果满足条件，尝试使模型预测失误
            image.requires_grad = False
            model.eval()

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)
                preds_str = preds_str[0][:preds_str[0].find('[s]')]

            if preds_str[0] != converter.decode(preds_index, length_for_pred)[0]:
                # 如果预测失误，保存扰动图像并停止
                return image, preds_str[0]

        # 根据图像质量调整alpha值
        if psnr_value < 40 or ssim_value < 0.9 or mse_value >= 100:
            alpha = max(min_alpha, alpha * decay_factor)
        else:
            alpha = min(max_alpha, alpha / decay_factor)

    return image, preds_str[0]


def predict(opt):
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt).to(device)
    print('Loading pretrained model from %s' % opt.saved_model)
    load_pretrained_model(model, opt.saved_model)
    model.eval()

    substitute_model = Model(opt).to(device)

    original_dataset = ImageFolderDataset(opt.image_folder, 32, 100)
    adversarial_dataset = ImageFolderDataset(opt.perturb_folder, 320, 1000)   #320 × 1000

    original_dataloader = DataLoader(original_dataset, batch_size=1, shuffle=False, num_workers=0)
    adversarial_dataloader = DataLoader(adversarial_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 训练替代模型
    train_substitute_model(substitute_model, model, original_dataloader, converter, opt, num_epochs=5)

    results = []
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_folder = f"{current_time}_{os.path.basename(opt.saved_model)}"
    perturbed_folder = os.path.join(output_folder, 'perturbed_images')
    downsampled_folder = os.path.join(output_folder, 'downsampled_images')

    os.makedirs(perturbed_folder, exist_ok=True)
    os.makedirs(downsampled_folder, exist_ok=True)


    success_count = 0
    total_count = 0

    original_preds_dict = {}

    for img, img_path in original_dataloader:
        img = img.to(device)
        original_img = img.clone()
        original_img.requires_grad = True

        if 'CTC' in opt.Prediction:
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(img, text_for_pred).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)])
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, preds_size)
        else:
            length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
            text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
            preds = model(img, text_for_pred, is_train=False)
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)
            preds_str = preds_str[0][:preds_str[0].find('[s]')]

        original_preds_dict[os.path.basename(img_path[0])] = preds_str[0]

    max_attempts = 50
    for img, img_path in adversarial_dataloader:
        img = img.to(device)
        original_img = img.clone()
        original_img.requires_grad = True

        attempts = 0
        success = False
        while attempts < max_attempts and not success:
            adversarial_img, adversarial_preds = adaptive_local_search_attack(original_img, substitute_model, converter,
                                                                              opt)
            adversarial_img_resized = transforms.Resize((32, 100))(adversarial_img)
            adversarial_img_resized = adversarial_img_resized.to(device)
            if 'CTC' in opt.Prediction:
                length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
                preds = model(adversarial_img_resized, text_for_pred).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)])
                _, preds_index = preds.max(2)
                adversarial_preds_str_resized = converter.decode(preds_index, preds_size)
            else:
                length_for_pred = torch.IntTensor([opt.batch_max_length]).to(device)
                text_for_pred = torch.LongTensor(1, opt.batch_max_length + 1).fill_(0).to(device)
                preds = model(adversarial_img_resized, text_for_pred, is_train=False)
                _, preds_index = preds.max(2)
                adversarial_preds_str_resized = converter.decode(preds_index, length_for_pred)
                adversarial_preds_str_resized = adversarial_preds_str_resized[0][
                                                :adversarial_preds_str_resized[0].find('[s]')]

            original_pred = original_preds_dict.get(os.path.basename(img_path[0]), "Unknown")
            adversarial_preds_str_resized = ''.join(adversarial_preds_str_resized)
            if original_pred != adversarial_preds_str_resized:
                success = True

            attempts += 1

        if success:
            adversarial_img_pil = transforms.ToPILImage()(adversarial_img.squeeze().cpu())
            adversarial_img_pil.save(os.path.join(perturbed_folder, os.path.basename(img_path[0])))

            adversarial_img_resized_pil = transforms.ToPILImage()(adversarial_img_resized.squeeze().cpu())
            adversarial_img_resized_pil.save(os.path.join(downsampled_folder, os.path.basename(img_path[0])))

            difference = cv2.absdiff(np.array(adversarial_img_pil), np.array(Image.open(img_path[0]).convert('RGB')))
            difference_pil = Image.fromarray(difference)

            results.append((img_path[0], original_pred, adversarial_preds_str_resized))
            success_count += 1

        total_count += 1

    consistent_count = 0
    inconsistent_count = 0

    for img_path, original_pred, adversarial_pred in results:
        consistency = "consistent" if original_pred == adversarial_pred else "inconsistent"
        if consistency == "consistent":
            consistent_count += 1
        else:
            inconsistent_count += 1
        print(
            f'Image: {img_path}, Original Prediction: {original_pred}, Adversarial Prediction: {adversarial_pred}, Consistency: {consistency}')

    successful_attempts = len(results)
    success_rate = (success_count / successful_attempts) * 100 if successful_attempts > 0 else 0

    print(f'Adversarial Attack Success Rate: {success_rate:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='Path to the folder containing images for prediction')
    parser.add_argument('--perturb_folder', required=True, help='Path to the folder containing images for perturbation')
    parser.add_argument('--saved_model', required=True, help='Path to the pretrained model')
    parser.add_argument('--batch_max_length', type=int, default=25, help='Maximum label length')
    parser.add_argument('--imgH', type=int, default=32, help='Height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='Width of the input image')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='Character label')
    parser.add_argument('--sensitive', action='store_false', help='Sensitive character mode')
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='Number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=3, help='Number of input channels of feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='Number of output channels of feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='Size of the LSTM hidden state')
    parser.add_argument('--targeted', action='store_true', help='Whether to perform targeted attacks')
    parser.add_argument('--target_class', type=int, help='Target class for targeted attack')

    opt = parser.parse_args()

    if opt.sensitive:
        opt.character = string.printable[:62]

    predict(opt)
