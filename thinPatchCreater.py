import numpy as np
import sys
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
from functions import caluculatePaddingSize, clipping, padding, getImageWithMeta, cropping

class ThinPatchCreater():
    def __init__(self, image, label, image_patch_width, label_patch_width, overlap=1, mask=None):
        """
        In 13 organs segmentation, 
        SimpleITK : [saggital, coronal, axial] ([x, y, z])
        numpy : [axial, coronal, saggital] ([z, y, x])

        This class use simpleITK method mainly and apply the method to images which are not consistent in spacing.
        """
        self.image = image
        self.label = label
        self.mask = mask
        self.image_patch_width = image_patch_width
        self.label_patch_width = label_patch_width
        self.z_slide = label_patch_width // overlap

        
    def execute(self):
        image_patch_size = np.array(self.image.GetSize())
        image_patch_size[2] = self.image_patch_width
        label_patch_size = np.array(self.label.GetSize())
        label_patch_size[2] = self.label_patch_width
        slide = np.array([0, 0, self.z_slide])

        image_size = np.array(self.image.GetSize())

        """ Caluculate padding size to clip the image correctly. """
        self.lower_pad_size, self.upper_pad_size = caluculatePaddingSize(
                image_size = image_size,
                image_patch = image_patch_size,
                label_patch = label_patch_size,
                slide = slide 
                )

        """ Pad image and label. """
        padded_image = padding(self.image, self.lower_pad_size[0].tolist(), self.upper_pad_size[0].tolist())
        # For restoration, add self..
        self.padded_label = padding(self.label, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())

        if self.mask is not None:
            padded_mask = padding(self.mask, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())

        """ Clip image, label and mask. """
        image_patch_list = self.makePatch(padded_image, self.image_patch_width, self.z_slide)
        label_patch_list = self.makePatch(self.padded_label, self.label_patch_width, self.z_slide)
        if self.mask is not None:
            mask_patch_list = self.makePatch(padded_mask, self.label_patch_width, self.z_slide)

        """ Check mask. """
        self.image_patch_list = []
        self.image_patch_array_list = []
        self.label_patch_list = []
        self.label_patch_array_list = []
        for i in range(len(image_patch_list)):
            if self.mask is not None:
                mask_patch_array = sitk.GetArrayFromImage(mask_patch_list[i])
                if (mask_patch_array == 0).all():
                    continue

            image_patch_array = sitk.GetArrayFromImage(image_patch_list[i])
            label_patch_array = sitk.GetArrayFromImage(label_patch_list[i])
            
            self.image_patch_list.append(image_patch_list[i])
            self.label_patch_list.append(label_patch_list[i])
            self.image_patch_array_list.append(image_patch_array)
            self.label_patch_array_list.append(label_patch_array)

    def output(self, kind):
        if kind == "Image":
            return self.image_patch_list, self.label_patch_list
        elif kind == "Array":
            return self.image_patch_array_list, self.label_patch_array_list
        else:
            print("[ERROR] Kind must be Array / Image.")
            sys.exit()

    def save(self, save_path):
        save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)

        with tqdm(total=len(self.image_patch_list), desc="Saving images...", ncols=60) as pbar:
            for i, (image_patch, label_patch) in enumerate(zip(self.image_patch_list, self.label_patch_list)):
                image_save_path = save_path / ("image_" + str(i).zfill(4) + ".mha")
                label_save_path = save_path / ("label_" + str(i).zfill(4) + ".mha")

                sitk.WriteImage(image_patch, str(image_save_path), True)
                sitk.WriteImage(label_patch, str(label_save_path), True)
                pbar.update(1)

    def restore(self, predicted_array_list):
        predicted_array = np.zeros(self.padded_label.GetSize()[::-1])
        z_size = self.padded_label.GetSize()[2] - self.label_patch_width
        patch_size = np.array(self.padded_label.GetSize())
        patch_size[2] = self.label_patch_width 

        indices = [ i for i in range(0, z_size + 1, self.z_slide)]
        with tqdm(total=len(indices), desc="Restoring image...", ncols=60) as pbar:
            for pre_array, index in zip(predicted_array_list, indices):
                z_slice = slice(index, index + patch_size[2])
                predicted_array[z_slice, ...] = pre_array
                
                pbar.update(1)

        predicted = getImageWithMeta(predicted_array, self.padded_label)

        predicted = cropping(predicted, self.lower_pad_size[1].tolist(), self.upper_pad_size[1].tolist())

        return predicted


    def makePatch(self, image, width, slide):
        patch_size = np.array(image.GetSize())
        patch_size[2] = width 

        z_size = image.GetSize()[2] - width

        indices = [ i for i in range(0, z_size + 1, slide)]
        patch_list = []
        with tqdm(total=len(indices), desc="Clipping images...", ncols=60) as pbar:
            for index in indices:
                lower_clip_size = np.array([0, 0, index])
                upper_clip_size = lower_clip_size + patch_size

                patch = clipping(image, lower_clip_size, upper_clip_size)

                patch_list.append(patch)

                pbar.update(1)

        return patch_list
