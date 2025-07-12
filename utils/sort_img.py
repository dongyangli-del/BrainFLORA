import os
from natsort import natsorted

def get_image_paths(folder_path):
    image_paths = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_paths.append(os.path.join(root, file))
    return natsorted(image_paths)

import cv2
import numpy as np

def image_histogram_similarity(image1_path, image2_path):
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity

def sort_images_by_similarity(folder1, folder2):
    images1 = get_image_paths(folder1)
    images2 = get_image_paths(folder2)
    
    if len(images1) != len(images2):
        raise ValueError("两个文件夹中的图片数量不一致")
    
    similarities = []
    for img1, img2 in zip(images1, images2):
        similarity = image_histogram_similarity(img1, img2)
        similarities.append((img1, img2, similarity))
    
    # 按相似度排序
    sorted_images = sorted(similarities, key=lambda x: x[2], reverse=True)
    return sorted_images

import shutil

def copy_sorted_images(sorted_images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for idx, (img1, img2, _) in enumerate(sorted_images):
        shutil.copy(img1, os.path.join(output_folder, f"img_{idx}_gen.jpg"))
        shutil.copy(img2, os.path.join(output_folder, f"img_{idx}_ori.jpg"))

def main():
    folder1 = "/mnt/dataset1/ldy/Workspace/FLORA/eval/fmri_generated_imgs/sub-03"
    # folder2 = "/mnt/dataset0/ldy/datasets/THINGS_EEG/images_set/test_images"
    folder2 = "/mnt/dataset0/ldy/datasets/fmri_dataset/images/test_images"
    output_folder = "/home/ldy/sub_03_sorted"
    
    sorted_images = sort_images_by_similarity(folder1, folder2)
    copy_sorted_images(sorted_images, output_folder)
    print(f"图片已按相似度排序并保存到 {output_folder}")

if __name__ == "__main__":
    main()