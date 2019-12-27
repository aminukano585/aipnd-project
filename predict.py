import argparse

import utilities


def main():
    parser = argparse.ArgumentParser(
        description='Image Classifier Command Line App\nPredict an Image'
    )
    
    parser.add_argument('img_path', type=str, default='flowers/test/1/image_06743.jpg',
                        help='Specify the path to the image to be classified e.g python predict.py input')
    parser.add_argument('checkpoint', type=str, default='checkpoints/checkpoint_v1.pth',
                        help='Specify the path to the saved checkpoint file e.g python predict.py input checkpoint')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Specify the number of top K most likely classes e.g --top_k 5')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json',
                        help='Specify the path to the categories mapping file e.g --category_names file')
    parser.add_argument('--gpu', action='store_true',
                        help='Specify whether to use GPU e.g --gpu')
    
    args = parser.parse_args()
    predict(args)
    
    
def predict(args):
    model, class_to_idx = utilities.load_checkpoint(args.checkpoint, args.gpu)
    probs, classes = utilities.get_prediction(args.img_path, model, args.top_k, args.gpu)
    
    classes = utilities.get_classes(classes, class_to_idx, args.category_names)
    
    utilities.display_result(probs, classes, args.img_path, args.category_names, args.checkpoint, args.top_k)


if __name__ == '__main__':
    main()