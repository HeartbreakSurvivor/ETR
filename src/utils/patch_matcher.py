import cv2
import numpy as np

class PatchMatcher(object):
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.match_method = config['match_method']
        self.total_points = config['total_points']
        self.ratio = 0.8

    def bf_match(self, q_kpts, db_kpts, q_desc, db_desc):
        print(' kp1, des1',  len(q_kpts), len(q_desc), q_desc[0])
        print(' kp2, des2',  len(db_kpts), len(db_desc))

        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(q_desc, db_desc, k=2)
        good = []
        good_points = []
        good_matches = []
        # print('type(raw_matches, shape', type(raw_matches), len(raw_matches))
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
                good.append(m1)

        print('good matches', len(good_points), good_points[0], good_matches[0])
        kpts1, kpts2 = [], []
        # put good points
        # kpts1 = np.float32([kp1[i].pt for (_, i) in good_points])
        # kpts2 = np.float32([kp2[i].pt for (i, _) in good_points])

        print('image1_kp', len(kpts1), kpts1)
        print('image2_kp', len(kpts2), kpts2)
        return kpts1, kpts2
        
    def match(self, q_kpts, db_kpts, q_desc=None, db_desc=None):
        if self.match_method == 'RANSAC':
            if not q_desc and not db_desc:
                return self.match_two_ransac(q_kpts, db_kpts)
            elif q_desc and db_desc:
                # use BFMatcher to find good matches
                q_kpts, db_kpts = self.bf_match(q_kpts, db_kpts, q_desc, db_desc)
                # 2. than match with keypoints use RANSAC
                return self.match_two_ransac(q_kpts, db_kpts)
        else:
            raise ValueError('unknown matcher descriptor')

    def match_two_ransac(self, q_kpts, db_kpts):
        assert len(q_kpts) == len(db_kpts), 'the length of keypoints must be same'
        
        # need at least four points to estimate a Homography
        if len(q_kpts) >= 4:
            _, mask = cv2.findHomography(srcPoints=q_kpts,
                                         dstPoints=db_kpts,
                                         method=cv2.RANSAC,
                                         ransacReprojThreshold=self.config['ransac_patch_thr'],
                                         maxIters=self.config['ransac_max_iters'],
                                         confidence=self.config['ransac_conf']
                                         )
            # RANSAC reproj threshold is set to the patch size whihc in image space for resnet-50 is 32

            # return inlier key points to draw matches for visualization
            inlier_query_kpts = q_kpts[mask.ravel() == 1]
            inlier_db_kpts = db_kpts[mask.ravel() == 1]
            
            inlier_count = inlier_query_kpts.shape[0]
            # score = inlier_count / self.total_points
            score = inlier_count / len(q_kpts)
            # bigger score means higher similarity
            return score, inlier_query_kpts, inlier_db_kpts
        else:
            return 0., None, None

class Image_Stitching():
    def __init__(self) :
        self.ratio=0.85
        self.min_match=10
        self.sift = cv2.xfeatures2d.SIFT_create()

    def registration(self,img1,img2):
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        print(' kp1, des1',  len(kp1), len(des1), des1[0])
        print(' kp2, des2',  len(kp2), len(des2))

        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        good = []
        good_points = []
        good_matches = []
        # print('type(raw_matches, shape', type(raw_matches), len(raw_matches))
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
                good.append(m1)
        print('good matches', len(good_points), good_points[0], good_matches[0])

        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None,
                            flags=0
                            )
        cv2.imwrite('matching_c.jpg', img3)

        if len(good_points) > self.min_match:
            image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])
            print('image1_kp', len(image1_kp), image1_kp)
            print('image2_kp', len(image2_kp), image2_kp)
            H, mask = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC)
        print('mask', len(mask), ' inliers:', sum(mask))
        
        img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, 
                            flags=2,
                            matchesMask = mask
                            )
        cv2.imwrite('matching_c2.jpg', img4)

        return H, mask

    def create_mask(self,img1,img2,version):
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        H, Status = self.registration(img1,img2)
        # print('Homogoraphy shape', H.shape, Status.shape)
        # print('H matrix', H)
        print('number of inliers', sum(Status))

        # height_img1 = img1.shape[0]
        # width_img1 = img1.shape[1]
        # width_img2 = img2.shape[1]
        # height_panorama = height_img1
        # width_panorama = width_img1 +width_img2

        # panorama1 = np.zeros((height_panorama, width_panorama, 3))
        # mask1 = self.create_mask(img1,img2,version='left_image')

        # panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        # panorama1 *= mask1
        # mask2 = self.create_mask(img1,img2,version='right_image')
        # panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        # result=panorama1+panorama2

        # rows, cols = np.where(result[:, :, 0] != 0)
        # min_row, max_row = min(rows), max(rows) + 1
        # min_col, max_col = min(cols), max(cols) + 1
        # final_result = result[min_row:max_row, min_col:max_col, :]
        # return final_result

if __name__ == '__main__':
    try: 
        # main(sys.argv[1],sys.argv[2])
        # main('./q11.jpg', './q22.jpg')
        # main('./images/left.jpeg', './images/right.jpeg')
        # main('./images/pair1.jpeg', './images/pair2.jpeg')
        # main('./images/d1.jpeg', './images/d2.jpeg')

        # main('./images/a.jpg', './images/b.jpg')
        main('./images/c1.jpg', './images/c2.jpg')

    except IndexError:
        print ("Please input two source images: ")
        print ("For example: python Image_Stitching.py '/Users/linrl3/Desktop/picture/p1.jpg' '/Users/linrl3/Desktop/picture/p2.jpg'")
    

