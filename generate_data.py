import argparse
from TFGeneration.GenerateTFRecord import *


def parse_args():
    parser = argparse.ArgumentParser(description='Tool to generate synthetic tables data.')
    parser.add_argument('--filesize', type=int, default=1, help='Number of images to store in one tfrecord. Default: 1.')
    parser.add_argument('--num_trecords', type=int, default=1000, help='Number of trecords files. Defult: 1000.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to run. More threads less time. Default: 1.')
    parser.add_argument('--outpath', type=str, default='tfrecords/', help='Output directory to store generated tfrecords. Default: tfrecords/.')
    parser.add_argument('--imagespath', default='../Table_Detection_Dataset/unlv/train/images', help='Directory containing UNLV dataset images.')
    parser.add_argument('--ocrpath', default='../Table_Detection_Dataset/unlv/unlv_xml_ocr', help='Directory containing ground truths of characters in UNLV dataset.')
    parser.add_argument('--tablepath', default='../Table_Detection_Dataset/unlv/unlv _xml_gt', help='Directory containing ground truths of tables in UNLV dataset.')
    parser.add_argument('--visualizeimgs', type=int, default=0, help='Store the generated images (along than tfrecords). Default: 0.')
    parser.add_argument('--visualizebboxes', type=int, default=0, help='Store the images with bound boxes. Default: 0.')
    return parser.parse_args()


def run_generator(args):
    filesize = max(int(args.filesize), 4)
    visualizeimgs = False
    if (args.visualizeimgs == 1):
        visualizeimgs = True

    visualizebboxes = False
    if (args.visualizebboxes == 1):
        visualizebboxes = True
    distributionfile = 'unlv_distribution'
    t = GenerateTFRecord(args.outpath, args.num_trecords, filesize, args.imagespath,
                         args.ocrpath, args.tablepath, visualizeimgs, visualizebboxes, distributionfile)
    t.write_to_tf(args.threads)


if __name__ == '__main__':
    args = parse_args()
    run_generator(args)
