import os
import sys
import json
import glob
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class ParkConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "bbox"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1  # HShin
    

    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 38  # Background + 40객체

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 30

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class ParkDataset(utils.Dataset):

    def load_park(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("bbox", 1, "Car")
        self.add_class("bbox", 2, "Van")
        self.add_class("bbox", 3, "Other Vehicle")
        self.add_class("bbox", 4, "Motorbike")
        self.add_class("bbox", 5, "Bicycle")
        self.add_class("bbox", 6, "Electric Scooter")
        self.add_class("bbox", 7, "Adult")
        self.add_class("bbox", 8, "Child")
        self.add_class("bbox", 9, "Stroller")
        self.add_class("bbox", 10, "Shopping Cart")
        self.add_class("bbox", 11, "Gate Arm")
        self.add_class("bbox", 12, "Parking Block")
        self.add_class("bbox", 13, "Speed Bump")
        self.add_class("bbox", 14, "Traffic Pole")
        self.add_class("bbox", 15, "Traffic Cone")
        self.add_class("bbox", 16, "Traffic Drum")
        self.add_class("bbox", 17, "Traffic Barricade")
        self.add_class("bbox", 18, "Cylindrical Bollard")
        self.add_class("bbox", 19, "U-shaped Bollard")
        self.add_class("bbox", 20, "Other Road Barriers")
        self.add_class("bbox", 21, "No Parking Stand")
        self.add_class("bbox", 22, "Adjustable Parking Pole")
        self.add_class("bbox", 23, "Waste Tire")
        self.add_class("bbox", 24, "Planter Barrier")
        self.add_class("bbox", 25, "Water Container")
        self.add_class("bbox", 26, "Movable Obstacle")
        self.add_class("bbox", 27, "Barrier Gate")
        self.add_class("bbox", 28, "Electric Car Charger")
        self.add_class("bbox", 29, "Parking Meter")
        self.add_class("bbox", 30, "Parking Sign")
        self.add_class("bbox", 31, "Traffic Light")
        self.add_class("bbox", 32, "Pedestrian Light")
        self.add_class("bbox", 33, "Street Sign")
        self.add_class("bbox", 34, "Disabled Parking Space")
        self.add_class("bbox", 35, "Pregnant Parking Space")
        self.add_class("bbox", 36, "Electric Car Parking Space")
        self.add_class("bbox", 37, "Two-wheeled Vehicle Parking Space")
        self.add_class("bbox", 38, "Other Parking Space")
        

        # Train or validation dataset?
        assert subset in ["Training", "Validation","test"]
#         label_dir = os.path.join(f"{dataset_dir}-{subset}","라벨링데이터")
#         data_dir = os.path.join(f"{dataset_dir}-{subset}","원천데이터")

        label_dir = os.path.join(dataset_dir, f"라벨링데이터_{subset}")
        data_dir = os.path.join(dataset_dir, f"원천데이터_{subset}")
        print(label_dir)
        print(data_dir)
        bbox_file_list = []
        annotations_b =[]

        # Load annotations
#         json_list = glob.glob(os.path.join(label_dir, "*.json"))
        json_list = glob.glob(os.path.join(label_dir, "*/*/*/label/*.json"))
        print(len(json_list))


        # Add images
        for a in json_list:
            with open(a, 'rb') as f:
                data = json.load(f)

                bboxs = [b['bbox'] for b in data['bbox2d']]
                name = [b['name'] for b in data['bbox2d']]
                name_dict ={
                    "Car" : 1,
                    "Van" : 2,
                    "Other Vehicle" : 3, 
                    "Motorbike" : 4,
                    "Bicycle" : 5,
                    "Electric Scooter" : 6,
                    "Adult" : 7,
                    "Child" : 8,
                    "Stroller" : 9,
                    "Shopping Cart" : 10,
                    "Gate Arm" : 11,
                    "Parking Block" : 12,
                    "Speed Bump" : 13,
                    "Traffic Pole" : 14,
                    "Traffic Cone" : 15,
                    "Traffic Drum" : 16,
                    "Traffic Barricade" : 17,
                    "Cylindrical Bollard" : 18,
                    "U-shaped Bollard" : 19,
                    "Other Road Barriers" : 20,
                    "No Parking Stand" : 21, 
                    "Adjustable Parking Pole" : 22,
                    "Waste Tire" : 23,
                    "Planter Barrier" : 24,
                    "Water Container" : 25,
                    "Movable Obstacle" : 26,
                    "Barrier Gate" : 27,
                    "Electric Car Charger" : 28,
                    "Parking Meter" : 29,
                    "Parking Sign" : 30,
                    "Traffic Light" : 31,
                    "Pedestrian Light" : 32,
                    "Street Sign" : 33,
                    "Disabled Parking Space" : 34,
                    "Pregnant Parking Space" : 35,
                    "Electric Car Parking Space" : 36,
                    "Two-wheeled Vehicle Parking Space" : 37,
                    "Other Parking Space" : 38
                    }

                num_ids = []

                for i,n in enumerate(name) : 
                    if n in name_dict :
                        num_ids.append(name_dict[n])
                    else : 
    #                     print(f'{a[0]} 파일 {n} 객체 오류')
                        del bboxs[i]

                a_img = a.split('/')
                file_name = f'{a_img[-1][:-5]}.jpg'
                image_path = os.path.join(data_dir,file_name)
                
                if os.path.exists(image_path):
                    try:
                        image = skimage.io.imread(image_path)
                        height, width = image.shape[:2]

                        self.add_image(
                            "bbox",
                            image_id=file_name,  # use file name as a unique image id
                            path=image_path,
                            width=width, height=height,
                            bboxs=bboxs,
                            num_ids=num_ids)
                        bbox_file_list.append(image_path)

                        if len(bbox_file_list)%1000==0:
                          print(len(bbox_file_list))
                    except :
    #                     print(f'{image_path} 불러오기 실패')
                        pass
        import csv
        
        with open(f'bboxlistfile{subset}.csv','w') as f :
            writer = csv.writer(f)
            writer.writerow(bbox_file_list)
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "bbox":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        if info["source"] != "bbox":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["bboxs"])],
                        dtype=np.uint8)
        if 'bboxs' in info :
            bboxs =info["bboxs"]

            for i in range(len(bboxs)) :
                bbox = bboxs[i]
                row_s, row_e = int(bbox[1]), int(bbox[3])
                col_s, col_e = int(bbox[0]), int(bbox[2])
                mask[row_s:row_e, col_s:col_e, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask.astype(np.bool_), num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bbox":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = ParkDataset()
    dataset_train.load_park(args.dataset, "Training")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = ParkDataset()
    dataset_val.load_park(args.dataset, "Validation")
    dataset_val.prepare()
    
    
    
    print("Training dataset Image Count: {}".format(len(dataset_train.image_ids)))
    print("Training dataset Class Count: {}".format(dataset_train.num_classes))
#     for i, info in enumerate(dataset_train.class_info):
#         print("{:3}. {:50}".format(i, info['name']))
        
    print("Validation dataset Image Count: {}".format(len(dataset_val.image_ids)))
    print("Validation dataset Class Count: {}".format(dataset_val.num_classes))
#     for i, info in enumerate(dataset_val.class_info):
#         print("{:3}. {:50}".format(i, info['name']))
        

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2

        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect park.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/dataset/",
                        help='Directory of the park dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = ParkConfig()
    else:
        class InferenceConfig(ParkConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 2
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))