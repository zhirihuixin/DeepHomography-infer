from collections import defaultdict
import cv2
import numpy as np
import imageio

from timer import Timer


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


if __name__ == "__main__":
    scale = 0.5
    MIN_MATCH_COUNT = 10
    img_1 = cv2.imread('../images/0000026_10028.jpg')
    img_2 = cv2.imread('../images/0000026_10001.jpg')

    img_1_ori = img_1.copy()
    img_2_ori = img_2.copy()

    h, w = img_1.shape[:2]
    img_1 = cv2.resize(img_1, (int(scale * w), int(scale * h)))
    img_2 = cv2.resize(img_2, (int(scale * w), int(scale * h)))

    img_1_gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_2_gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    # feature2D = cv2.SIFT_create(100)
    feature2D = cv2.ORB_create(100, 2.0)

    timers = defaultdict(Timer)

    kp2, des2 = feature2D.detectAndCompute(img_2_gray, None)
    for i in range(1000):
        timers['all_time'].tic()
        timers['feature2D'].tic()
        # find the keypoints and descriptors with feature2D
        kp1, des1 = feature2D.detectAndCompute(img_1_gray, None)
        timers['feature2D'].toc()

        timers['matches'].tic()
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(des1, des2, k=2)

        matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
        timers['matches'].toc()

        timers['post'].tic()
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if scale != 1.0:
                H[0, 2] /= scale
                H[1, 2] /= scale
                H[2, 0] *= scale
                H[2, 1] *= scale
            # im_out = cv2.warpPerspective(
            #     img_1_ori, H, (img_1_ori.shape[1], img_1_ori.shape[0]), borderValue=(255, 255, 255)
            # )
            im_out = cv2.warpPerspective(
                img_1_ori, H, (img_1_ori.shape[1], img_1_ori.shape[0])
            )
        timers['post'].toc()
        timers['all_time'].toc()

        for k, v in timers.items():
            if k != 'all_time':
                print(' | {}: {:.3f}s'.format(k, v.average_time))
        print(' ------| {}: {:.3f}s'.format('all_time', timers['all_time'].average_time))
    
        cv2.imwrite('tmp/test.jpg', im_out)
        img_1_ori = cv2.cvtColor(img_1_ori, cv2.COLOR_BGR2RGB)
        img_2_ori = cv2.cvtColor(img_2_ori, cv2.COLOR_BGR2RGB)
        im_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2RGB)
        input_list = [img_1_ori, img_2_ori]
        output_list = [im_out, img_2_ori]
        change_list = [im_out, img_1_ori]
        create_gif(input_list, "tmp/input.gif")
        create_gif(output_list, "tmp/output.gif")
        create_gif(change_list, "tmp/change.gif")
        break