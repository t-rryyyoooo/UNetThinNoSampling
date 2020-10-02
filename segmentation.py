import SimpleITK as sitk
import numpy as np
import argparse
from functions import createParentPath, getImageWithMeta, croppingForNumpy
from pathlib import Path
from thinPatchCreater import ThinPatchCreater 
from tqdm import tqdm
import torch
import cloudpickle
import re


def ParseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", help="$HOME/Desktop/data/kits19/case_00000/imaging.nii.gz")
    parser.add_argument("modelweightfile", help="Trained model weights file (*.pkl).")
    parser.add_argument("save_path", help="Segmented label file.(.mha)")
    parser.add_argument("--image_patch_width", default=8, type=int)
    parser.add_argument("--label_patch_width", default=8, type=int)
    parser.add_argument("--overlap", default=1, type=int)
    parser.add_argument("-g", "--gpuid", help="0 1", nargs="*", default=0, type=int)

    args = parser.parse_args()
    return args

def main(args):
    """ Slice module. """
    image = sitk.ReadImage(args.image_path)
    dummy = sitk.Image(image.GetSize(), sitk.sitkUInt8)

    tpc = ThinPatchCreater(
            image = image,
            label = dummy,
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            overlap = args.overlap,
            )

    tpc.execute()
    image_array_list, _ = tpc.output(kind="Array")

    use_cuda = torch.cuda.is_available() and True
    device = torch.device("cuda" if use_cuda else "cpu")
    """ Load model. """
    with open(args.modelweightfile, 'rb') as f:
        model = cloudpickle.load(f)
        model = torch.nn.DataParallel(model, device_ids=args.gpuid)

    model.eval()

    """ Segmentation module. """
    segmented_array_list = []
    for image_array in tqdm(image_array_list, desc="Segmenting images...", ncols=60):
        image_array = torch.from_numpy(image_array)[None, None, ...].to(device, dtype=torch.float)

        segmented_array = model(image_array)
        segmented_array = segmented_array.to("cpu").detach().numpy().astype(np.float)
        segmented_array = np.squeeze(segmented_array)
        segmented_array = np.argmax(segmented_array, axis=0).astype(np.uint8)

        diff = args.image_patch_width - args.label_patch_width
        lower_crop_size = [diff // 2, 0, 0]
        upper_crop_size = [diff // 2, 0, 0]
        segmented_array = croppingForNumpy(segmented_array, lower_crop_size, upper_crop_size)
        segmented_array_list.append(segmented_array)

    """ Restore module. """
    segmented = tpc.restore(segmented_array_list)

    createParentPath(args.save_path)
    print("Saving image to {}".format(args.save_path))
    sitk.WriteImage(segmented, args.save_path, True)


if __name__ == '__main__':
    args = ParseArgs()
    main(args)
    
