"""
Classifier tối ưu với các cải tiến:
1. Data augmentation khi training
2. Validation set để đánh giá
3. Tối ưu hyperparameters
4. Cross-validation
5. Better preprocessing
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def augment_image(image):
    """
    Tối ưu: Data augmentation để tăng độ đa dạng dữ liệu
    """
    # Random brightness
    if np.random.random() > 0.5:
        brightness = np.random.uniform(0.8, 1.2)
        image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    
    # Random contrast
    if np.random.random() > 0.5:
        contrast = np.random.uniform(0.8, 1.2)
        mean = np.mean(image)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
    
    return image

def load_data_with_augmentation(image_paths, do_augment=False, image_size=160):
    """
    Tối ưu: Load data với augmentation option
    """
    nrof_samples = len(image_paths)
    images = np.zeros((nrof_samples, image_size, image_size, 3))
    
    for i in range(nrof_samples):
        import imageio
        img = imageio.imread(image_paths[i])
        if img.ndim == 2:
            img = facenet.to_rgb(img)
        
        # Augmentation trước khi prewhiten
        if do_augment:
            img = augment_image(img)
        
        # Prewhiten
        img = facenet.prewhiten(img)
        
        # Crop và flip (giống như training FaceNet)
        img = facenet.crop(img, False, image_size)
        img = facenet.flip(img, False)
        
        images[i,:,:,:] = img
    
    return images

def main(args):
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            np.random.seed(seed=args.seed)
            
            # Load dataset
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(
                    dataset_tmp, 
                    args.min_nrof_images_per_class, 
                    args.nrof_train_images_per_class
                )
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert len(cls.image_paths) > 0, 'There must be at least one image for each class in the dataset'

            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            # Tối ưu: Filter chỉ lấy file ảnh (loại bỏ .txt, .json, etc.)
            valid_indices = [i for i, p in enumerate(paths) 
                           if p.lower().endswith(('.jpg', '.jpeg', '.png'))]
            paths = [paths[i] for i in valid_indices]
            labels = [labels[i] for i in valid_indices]
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Tối ưu: Split train/validation nếu mode TRAIN
            if args.mode == 'TRAIN' and args.use_validation:
                train_paths, val_paths, train_labels, val_labels = train_test_split(
                    paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
                )
                print(f'Train images: {len(train_paths)}, Validation images: {len(val_paths)}')
            else:
                train_paths, train_labels = paths, labels
                val_paths, val_labels = None, None
            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(train_paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = train_paths[start_index:end_index]
                
                # Tối ưu: Sử dụng augmentation khi training
                use_augment = (args.mode == 'TRAIN' and args.use_augmentation)
                images = load_data_with_augmentation(paths_batch, do_augment=use_augment, image_size=args.image_size)
                
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            # Tối ưu: Normalize embeddings (cải thiện SVM performance)
            if args.normalize_embeddings:
                scaler = StandardScaler()
                emb_array = scaler.fit_transform(emb_array)
                print('[OK] Normalized embeddings')
            
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Tối ưu: Train classifier với các options
                print('Training classifier')
                
                # Tối ưu: Chọn kernel và C tốt hơn
                if args.svm_kernel == 'rbf':
                    # RBF kernel tốt hơn cho dữ liệu phức tạp
                    gamma_val = args.svm_gamma
                    if gamma_val not in ['scale', 'auto']:
                        try:
                            gamma_val = float(gamma_val)
                        except ValueError:
                            gamma_val = 'scale'
                    model = SVC(
                        kernel='rbf', 
                        probability=True, 
                        C=args.svm_c,
                        gamma=gamma_val
                    )
                else:
                    # Linear kernel nhanh hơn, tốt cho dữ liệu đã được normalize
                    model = SVC(
                        kernel='linear', 
                        probability=True,
                        C=args.svm_c
                    )
                
                print(f'Training with kernel={args.svm_kernel}, C={args.svm_c}')
                model.fit(emb_array, train_labels)
                
                # Tối ưu: Cross-validation để đánh giá
                if args.use_cross_validation:
                    print('Running cross-validation...')
                    cv_scores = cross_val_score(model, emb_array, train_labels, cv=min(5, len(dataset)))
                    print(f'Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
                
                # Tối ưu: Validate trên validation set
                if val_paths is not None and len(val_paths) > 0:
                    print('Calculating validation embeddings...')
                    nrof_val_images = len(val_paths)
                    nrof_val_batches = int(math.ceil(1.0*nrof_val_images / args.batch_size))
                    val_emb_array = np.zeros((nrof_val_images, embedding_size))
                    
                    for i in range(nrof_val_batches):
                        start_index = i*args.batch_size
                        end_index = min((i+1)*args.batch_size, nrof_val_images)
                        paths_batch = val_paths[start_index:end_index]
                        images = load_data_with_augmentation(paths_batch, do_augment=False, image_size=args.image_size)
                        
                        feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                        val_emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                    
                    if args.normalize_embeddings:
                        val_emb_array = scaler.transform(val_emb_array)
                    
                    val_predictions = model.predict(val_emb_array)
                    val_accuracy = np.mean(np.equal(val_predictions, val_labels))
                    print(f'Validation accuracy: {val_accuracy:.4f}')
                    
                    # Classification report
                    print('\nValidation Classification Report:')
                    class_names = [cls.name.replace('_', ' ') for cls in dataset]
                    print(classification_report(val_labels, val_predictions, target_names=class_names))
            
                # Create a list of class names
                class_names = [cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                save_data = {
                    'model': model,
                    'class_names': class_names
                }
                
                # Tối ưu: Lưu thêm scaler nếu dùng normalization
                if args.normalize_embeddings:
                    save_data['scaler'] = scaler
                
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump(save_data, outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            elif (args.mode=='CLASSIFY'):
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    loaded_data = pickle.load(infile)
                
                # Tối ưu: Hỗ trợ cả format cũ và mới
                if isinstance(loaded_data, tuple):
                    model, class_names = loaded_data
                    scaler = None
                else:
                    model = loaded_data['model']
                    class_names = loaded_data['class_names']
                    scaler = loaded_data.get('scaler', None)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                # Normalize nếu có scaler
                if scaler is not None:
                    emb_array = scaler.transform(emb_array)
                    print('[OK] Applied normalization from saved scaler')

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    # Tối ưu: Thêm các options mới
    parser.add_argument('--use_augmentation', 
        help='Use data augmentation when training', action='store_true')
    parser.add_argument('--use_validation', 
        help='Split data into train/validation sets', action='store_true')
    parser.add_argument('--normalize_embeddings', 
        help='Normalize embeddings before training (improves SVM performance)', action='store_true')
    parser.add_argument('--use_cross_validation', 
        help='Run cross-validation to evaluate model', action='store_true')
    parser.add_argument('--svm_kernel', type=str, choices=['linear', 'rbf'], default='linear',
        help='SVM kernel type')
    parser.add_argument('--svm_c', type=float, default=1.0,
        help='SVM regularization parameter C')
    parser.add_argument('--svm_gamma', type=str, default='scale',
        help='SVM gamma parameter for RBF kernel (float or "scale" or "auto")')
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

