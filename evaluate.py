import tensorflow as tf
import os
import json
import subprocess
from scipy.misc import imread, imresize
from scipy import misc

from train import build_forward
from utils.annolist import AnnotationLib as al
from utils.train_utils import add_rectangles, rescale_boxes

import cv2
import argparse

def get_image_dir(args):
    weights_iteration = int(args.weights.split('-')[-1])
    expname = '_' + args.expname if args.expname else ''
    image_dir = '%s/images_%s_%d%s' % (os.path.dirname(args.weights), os.path.basename(args.test_boxes)[:-5], weights_iteration, expname)
    return image_dir

def get_results(args, H):
    tf.reset_default_graph()
    x_in = tf.placeholder(tf.float32, name='x_in', shape=[H['image_height'], H['image_width'], 3])
    p1_x_in = tf.placeholder(tf.float32, name='p1_x_in', shape=[H['image_height'], H['image_width'], 3])
    p2_x_in = tf.placeholder(tf.float32, name='p2_x_in', shape=[H['image_height'], H['image_width'], 3])
    p3_x_in = tf.placeholder(tf.float32, name='p3_x_in', shape=[H['image_height'], H['image_width'], 3])
    p4_x_in = tf.placeholder(tf.float32, name='p4_x_in', shape=[H['image_height'], H['image_width'], 3])
    p5_x_in = tf.placeholder(tf.float32, name='p5_x_in', shape=[H['image_height'], H['image_width'], 3])
    p6_x_in = tf.placeholder(tf.float32, name='p6_x_in', shape=[H['image_height'], H['image_width'], 3])
    p7_x_in = tf.placeholder(tf.float32, name='p7_x_in', shape=[H['image_height'], H['image_width'], 3])
    p8_x_in = tf.placeholder(tf.float32, name='p8_x_in', shape=[H['image_height'], H['image_width'], 3])
    f_x_in = tf.placeholder(tf.float32, name='f_x_in', shape=[H['image_height'], H['image_width'], 3])

    if H['use_rezoom']:
        pred_boxes, pred_logits, pred_confidences, pred_confs_deltas, pred_boxes_deltas = build_forward(H, tf.expand_dims(x_in, 0), 
                                                                                                           tf.expand_dims(p1_x_in, 0), 
                                                                                                           tf.expand_dims(p2_x_in, 0),
                                                                                                           tf.expand_dims(p3_x_in, 0),
                                                                                                           tf.expand_dims(p4_x_in, 0),
                                                                                                           tf.expand_dims(p5_x_in, 0),
                                                                                                           tf.expand_dims(p6_x_in, 0),
                                                                                                           tf.expand_dims(p7_x_in, 0),
                                                                                                           tf.expand_dims(p8_x_in, 0),
                                                                                                           tf.expand_dims(f_x_in, 0),
                                                                                                           'test', reuse=None)
        grid_area = H['grid_height'] * H['grid_width']
        pred_confidences = tf.reshape(tf.nn.softmax(tf.reshape(pred_confs_deltas, [grid_area * H['rnn_len'], 2])), [grid_area, H['rnn_len'], 2])
        if H['reregress']:
            pred_boxes = pred_boxes + pred_boxes_deltas
    else:
        pred_boxes, pred_logits, pred_confidences = build_forward(H, tf.expand_dims(x_in, 0), 'test', reuse=None)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver.restore(sess, args.weights)

        pred_annolist = al.AnnoList()

        true_annolist = al.parse(args.test_boxes)
        data_dir = os.path.dirname(args.test_boxes)
        image_dir = get_image_dir(args)
        subprocess.call('mkdir -p %s' % image_dir, shell=True)
        for i in range(len(true_annolist)):
            true_anno = true_annolist[i]
            orig_img = imread('%s/%s' % (data_dir, true_anno.imageName))[:,:,:3]
            dir_path = os.path.dirname(true_anno.imageName)
            file_name = true_anno.imageName.split('/')[-1]
            (shotname, extension) = os.path.splitext(file_name)
            p1_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 1)).zfill(4) + ".png"
            p2_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 2)).zfill(4) + ".png"
            p3_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 3)).zfill(4) + ".png"
            p4_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 4)).zfill(4) + ".png"
            p5_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 5)).zfill(4) + ".png"
            p6_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 6)).zfill(4) + ".png"
            p7_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 7)).zfill(4) + ".png"
            p8_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) - 8)).zfill(4) + ".png"
            f_image_path = data_dir + "/" + dir_path + "/" + (str(int(shotname) + 1)).zfill(4) + ".png"
            if not os.path.exists(p1_image_path):
                print "File not exists: %s" % p1_image_path
                exit()
            if not os.path.exists(p2_image_path):
                print "File not exists: %s" % p2_image_path
                exit()
            if not os.path.exists(f_image_path):
                print "File not exists: %s" % f_image_path
                exit()

            p1_img = imread(p1_image_path)
            p2_img = imread(p2_image_path)
            p3_img = imread(p3_image_path)
            p4_img = imread(p4_image_path)
            p5_img = imread(p5_image_path)
            p6_img = imread(p6_image_path)
            p7_img = imread(p7_image_path)
            p8_img = imread(p8_image_path)
            f_img = imread(f_image_path)

            img = imresize(orig_img, (H["image_height"], H["image_width"]), interp='cubic')
            feed = {x_in: img, p1_x_in: p1_img, p2_x_in: p2_img, p3_x_in: p3_img, p4_x_in: p4_img, p5_x_in: p5_img, p6_x_in: p6_img, p7_x_in: p7_img, p8_x_in: p8_img, f_x_in: f_img}
            (np_pred_boxes, np_pred_confidences) = sess.run([pred_boxes, pred_confidences], feed_dict=feed)
            pred_anno = al.Annotation()
            pred_anno.imageName = true_anno.imageName
            new_img, rects = add_rectangles(H, [img], np_pred_confidences, np_pred_boxes,
                                            use_stitching=True, rnn_len=H['rnn_len'], min_conf=args.min_conf, tau=args.tau, show_suppressed=args.show_suppressed)
        
            pred_anno.rects = rects
            pred_anno.imagePath = os.path.abspath(data_dir)
            pred_anno = rescale_boxes((H["image_height"], H["image_width"]), pred_anno, orig_img.shape[0], orig_img.shape[1])
            pred_annolist.append(pred_anno)
            
            imname = '%s/%s' % (image_dir, os.path.basename(true_anno.imageName))
            misc.imsave(imname, new_img)
            if i % 25 == 0:
                print(i)
    return pred_annolist, true_annolist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--expname', default='')
    parser.add_argument('--test_boxes', required=True)
    parser.add_argument('--gpu', default=1)
    parser.add_argument('--logdir', default='output')
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--tau', default=0.25, type=float)
    parser.add_argument('--min_conf', default=0.1, type=float)
    parser.add_argument('--show_suppressed', default=True, type=bool)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    hypes_file = '%s/hypes.json' % os.path.dirname(args.weights)
    with open(hypes_file, 'r') as f:
        H = json.load(f)
    expname = args.expname + '_' if args.expname else ''
    pred_boxes = '%s.%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))
    true_boxes = '%s.gt_%s%s' % (args.weights, expname, os.path.basename(args.test_boxes))


    pred_annolist, true_annolist = get_results(args, H)
    pred_annolist.save(pred_boxes)
    true_annolist.save(true_boxes)

    try:
        rpc_cmd = './utils/annolist/doRPC.py --minOverlap %f %s %s' % (args.iou_threshold, true_boxes, pred_boxes)
        print('$ %s' % rpc_cmd)
        rpc_output = subprocess.check_output(rpc_cmd, shell=True)
        print(rpc_output)
        txt_file = [line for line in rpc_output.split('\n') if line.strip()][-1]
        output_png = '%s/results.png' % get_image_dir(args)
        plot_cmd = './utils/annolist/plotSimple.py %s --output %s' % (txt_file, output_png)
        print('$ %s' % plot_cmd)
        plot_output = subprocess.check_output(plot_cmd, shell=True)
        print('output results at: %s' % plot_output)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
