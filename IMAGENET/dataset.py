
import tensorflow as tf
import os
import math
from abstract_dataset import Abstract_Dataset


class Dataset(Abstract_Dataset):
    def __init__(self, opts, data_type):
        Abstract_Dataset.__init__(self, opts, data_type)        
        self.datapath = os.path.join(opts.data, data_type)



    def init_metadata(self, data_type):
        self.input_depth = 3
        self.input_height = 224
        self.input_width = 224
        self.image_size = 256
        if data_type == 'train':
            self.size =  1200000 
        else:
            self.size = 50000


    def file_list_from_txt(self, data_dir):
        dir_txt = data_dir + ".txt"
        filenames = []
        with open(dir_txt, 'r') as f:
            for line in f:
                if line[0] == '.': continue
                line = line.rstrip()
                fn = os.path.join(data_dir, line)
                filenames.append(fn)
        return filenames


    def file_list(self, data_dir):
        filenames = []
        counter = 0
        for dir_ in os.listdir(data_dir):
            for file_ in os.listdir(os.path.join(data_dir, dir_)):
                if os.path.isfile(os.path.join(data_dir, dir_, file_)):
                    line = file_.rstrip()
                    fn = os.path.join(data_dir, dir_, line)
                    filenames.append(fn)
                    counter += 1
        return filenames, counter


    def load_data(self, data_dir):
        data = []
        i = 0
        print("-- listing files in", data_dir)
        start_time = time.time()
        files, self.image_number = self.file_list(data_dir)
        for img_fn in files:
            ext = os.path.splitext(img_fn)[1]
            if ext != '.JPEG': self.image_number -= 1
            label_name = re.search(r'(n\d+)', img_fn).group(1)
            fn = os.path.join(data_dir, img_fn)
            label_index = synset_map[label_name]["index"]
            data.append({
                "filename": fn,
                "label_name": label_name,
                "label_index": label_index,
                "desc": synset[label_index],
            })
        duration = time.time() - start_time
        print ("## took %f sec, load %d filenames for %s dataset" % (duration, self.image_number, self.data_type))
        return data




    def build_input(self):
        data = self.load_data(self.datapath)
        filenames = [ d['filename'] for d in data ]
        label_indexes = [ d['label_index'] for d in data ]
        # Up to here, nothing is symbolic
        filename, label_index = tf.train.slice_input_producer([filenames, label_indexes], shuffle = (self.data_type == 'train'))#not (self.data_type == 'val'))
        image_file = tf.read_file(filename)
        image_data = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        image = self.image_preprocessing(image_data)
        image = tf.transpose(image, [2, 0, 1])    
        return image, [label_index]

        

    def image_preprocessing_(self, image):
        im = image
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        im = tf.image.resize_images(im, [self.input_size, self.input_size], method=tf.image.ResizeMethod.BILINEAR)
        im = im - mean
        im = im / std
        if self.data_type =='train':
            im = tf.image.random_flip_left_right(im)
        return im


    def image_preprocessing(self, image):        
        if self.data_type =='train':
            image = self.distort_image(image)
        elif self.data_type =='val':
            image = eval_image(image)
        else:
            image = eval_image(image)
        return image



    def distort_color(self, image):
        color_ordering=0
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        elif color_ordering == 1:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
        else:
            pass
        return image


    def distort_image(self, image):
        shape = tf.shape(image)
        min_scale = tf.random_uniform([], minval=256, maxval=480, dtype=tf.float32 )
        max_ratio = tf.cast(shape[1], tf.float32) / tf.cast(shape[0], tf.float32)
        scaled_height, scaled_width = control_flow_ops.cond(shape[0] < shape[1], lambda: (min_scale, min_scale*max_ratio ), lambda: (min_scale/max_ratio , min_scale))
        distorted_image = image
        distorted_image = tf.expand_dims(distorted_image, 0)
        distorted_image = tf.image.resize_bilinear(distorted_image, [tf.cast(scaled_height, tf.int32), tf.cast(scaled_width, tf.int32)], align_corners=False)
        distorted_image = tf.squeeze(distorted_image, [0])
        distorted_image = tf.random_crop(distorted_image, [self.input_height, self.input_width, self.input_depth])    
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        # Randomly distort the colors.
        distorted_image = self.distort_color(distorted_image)
        return distorted_image


    def eval_image_(image, height, width, scope=None):
        shape = tf.shape(image)
        short_edge = tf.reduce_min(shape[:2])
        #short_edge = tf.Print(short_edge, [short_edge, tf.shape(image)[0], tf.shape(image)[1]])
        crop_img = tf.image.resize_image_with_crop_or_pad(image, short_edge, short_edge)
        resized_img = tf.image.resize_images(crop_img, [height, width], method=tf.image.ResizeMethod.BILINEAR)
        return resized_img


    def eval_image(self, image):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225] 
        im = image
        shape = tf.shape(im)
        height = tf.cast(shape[0], tf.float32)
        width = tf.cast(shape[1], tf.float32)
        height_smaller_than_width = tf.less_equal(height, width)
        new_shorter_edge = tf.constant(self.image_size)
        new_height, new_width = control_flow_ops.cond(
            height_smaller_than_width,
            lambda: (new_shorter_edge, new_shorter_edge * width / height ),
            lambda: (new_shorter_edge * height / width , new_shorter_edge))
        im = tf.image.resize_images(im, [tf.cast(new_height, tf.int32), tf.cast(new_width, tf.int32)], method=tf.image.ResizeMethod.BILINEAR)
        im = tf.image.resize_image_with_crop_or_pad(im, self.input_height, self.input_width)
        im = im - mean
        im = im / std
        return im


