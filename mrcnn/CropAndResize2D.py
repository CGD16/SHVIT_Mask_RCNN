import tensorflow as tf


class CropAndResize2D:
    def __init__(self, method_name = 'bilinear', extrapolation_value = 0.0):
        if method_name not in ['bilinear', 'nearest']:
            raise ValueError("Method must be 'bilinear' or 'nearest'")
        self.method_name = method_name
        self.extrapolation_value = extrapolation_value
        self.img = tf.TensorArray(tf.int32, size=1, dynamic_size=True)
        
    @tf.function
    def call(self, image, boxes, box_indices, crop_size):
        if len(image.shape) != 4:
            raise ValueError(f"Input image must be 4-D, but got shape: {image.shape}")
        # batch_size, image_height, image_width, depth = image.shape
        # if image_height <= 0 or image_width <= 0:
        #   raise ValueError("Image dimensions must be positive")
        num_boxes, box_index = None, None
        shape_boxes = tf.shape(boxes)
        shape_box_indices = tf.shape(box_indices)
            
        if shape_boxes[0] == 0 and shape_box_indices[0] == 0:
            num_boxes = 0
            box_indices = tf.constant([], dtype=tf.int32)
        else:
            num_boxes = shape_boxes[0]
        # if len(crop_size) != 2:
        #   raise ValueError(f"Crop size must be a list of 2 elements, but got: {crop_size}")
        crop_height, crop_width = crop_size
        # if crop_height <= 0 or crop_width <= 0:
        #   raise ValueError("Crop dimensions must be positive")
        crops = []
        for i in range(num_boxes):
            box = boxes[i]
            box_index = box_indices[i]
            print(box_index)
            img = image[box_index]
            ##
            y1, x1, y2, x2 = box[0], box[1], box[2], box[3]
            image_height, image_width, channels = img.shape[0], img.shape[1], img.shape[2]

            print("X"*20)
            print("y1", y1)
            print("image_height", image_height)
            # Calculate bounding box coordinates
            top = tf.cast(y1 * image_height, tf.int32)
            left = tf.cast(x1 * image_width, tf.int32)
            bottom = tf.cast(y2 * image_height, tf.int32)
            right = tf.cast(x2 * image_width, tf.int32)
            print("Y"*20)

            # Ensure bounding box coordinates are within image dimensions
            top = tf.clip_by_value(top, 0, tf.cast(image_height, tf.int32) - 1)
            left = tf.clip_by_value(left, 0, tf.cast(image_width, tf.int32) - 1)
            bottom = tf.clip_by_value(bottom, 0, tf.cast(image_height, tf.int32) - 1)
            right = tf.clip_by_value(right, 0, tf.cast(image_width, tf.int32) - 1)
                
            # Calculate height and width of the box
            box_height = bottom - top
            box_width = right - left
            # if box_height <= 0 or box_width <= 0:
            #   raise ValueError(f"Invalid box dimensions: height={box_height}, width={box_width}")

            # Crop the image
            cropped_image = tf.image.crop_to_bounding_box(img, top, left, box_height, box_width)
                
            # Resize the cropped image
            crop = tf.image.resize(cropped_image, [crop_height, crop_width], method='bilinear') # self.method_name)
            crops.append(crop)
         
        return tf.stack(crops)