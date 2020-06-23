import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from .models.keras_ssd import ssd_512
from .keras_loss_function.keras_ssd_loss import SSDLoss
from skimage.transform import resize
import cv2
# tf.keras.backend.clear_session()  # Clear previous models from memory.


def build_ssd512(img_height=512, img_width=512):
    model = ssd_512(image_size=(img_height, img_width, 3),
                    n_classes=20,
                    mode='inference',
                    l2_regularization=0.0005,
                    scales=[0.07, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.05],
                    aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5, 3.0, 1.0 / 3.0],
                                             [1.0, 2.0, 0.5],
                                             [1.0, 2.0, 0.5]],
                    two_boxes_for_ar1=True,
                    steps=[8, 16, 32, 64, 128, 256, 512],
                    offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                    clip_boxes=False,
                    variances=[0.1, 0.1, 0.2, 0.2],
                    normalize_coords=True,
                    subtract_mean=[123, 117, 104],
                    swap_channels=[2, 1, 0],
                    confidence_thresh=0.5,
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400)

    # 2: Load the trained weights into the model.

    # TODO: Set the path of the trained weights.
    weights_path = './weights_of_neural_networks/weights_of_trained_ssd/VGG_VOC0712_SSD_512x512_iter_120000.h5'
    model.load_weights(weights_path, by_name=True)
    adam = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
    model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
    return model


def ssd_inference(model, orig_image, confidence_threshold=0.5):
    input_image = resize(orig_image, (512, 512), order=1) * 255
    # cv2.imshow('hello', orig_image)
    # cv2.waitKey(0)
    y_pred = model.predict(np.array([input_image]))
    y_pred_thresh = [y_pred[k][y_pred[k, :, 1] > confidence_threshold] for k in range(y_pred.shape[0])]
    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    # print("Predicted boxes:\n")
    # print('   class   conf xmin   ymin   xmax   ymax')
    # print(y_pred_thresh[0])

    # Display the image and draw the predicted boxes onto it.

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    orig_image = np.array(orig_image)
    for box in y_pred_thresh[0]:
        # Transform the predicted bounding boxes for the 512x512 image to the original image dimensions.
        xmin = box[-4] * orig_image.shape[1] / 512
        ymin = box[-3] * orig_image.shape[0] / 512
        xmax = box[-2] * orig_image.shape[1] / 512
        ymax = box[-1] * orig_image.shape[0] / 512
        orig_image = cv2.rectangle(orig_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 4)
        color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        orig_image = cv2.putText(
            orig_image, label, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    return orig_image
