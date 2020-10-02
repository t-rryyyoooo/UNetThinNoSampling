import argparse
import SimpleITK as sitk
from thinPatchCreater import ThinPatchCreater

def parseArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument("image_path")
    parser.add_argument("label_path")
    parser.add_argument("save_path")
    parser.add_argument("--image_patch_width", default=8, type=int)
    parser.add_argument("--label_patch_width", default=8, type=int)
    parser.add_argument("--overlap", default=1, type=int)
    parser.add_argument("--mask_path")

    args = parser.parse_args()

    return args

def main(args):
    image = sitk.ReadImage(args.image_path)
    label = sitk.ReadImage(args.label_path)
    
    if args.mask_path is not None:
        mask = sitk.ReadImage(args.mask_path)
    else:
        mask = None

    tpc = ThinPatchCreater(
            image = image,
            label = label, 
            image_patch_width = args.image_patch_width,
            label_patch_width = args.label_patch_width,
            overlap = args.overlap,
            mask = mask
            )

    tpc.execute()
    """
    image_array_list, label_array_list = tpc.output(kind="Array")
    pre = tpc.restore(label_array_list)
    from functions import DICE
    a = sitk.GetArrayFromImage(label)
    b = sitk.GetArrayFromImage(pre)
    dice = DICE(a, b)
    print(dice)
    """
    tpc.save(args.save_path)



if __name__ == "__main__":
    args = parseArgs()
    main(args)
