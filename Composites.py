from __future__ import division, print_function
import cv2
import numpy as np
import os
import random
import imghdr

class Composites:
    def __init__(self,filedir=None,filelist='mirflickr_filelist.txt',seed=1723,sample_size = 5000):
        #self.rand_images = [cv2.imread(l.rstrip()) for i,l in enumerate(open(filelist,'r')) if i < 500]
        if not filedir is None:
            filenames = [l for l in self.load_filenames_from_dir(filedir)]
        else:
            filenames = [l.rstrip() for l in open(filelist,'r')]
        random.seed(seed)
        random.shuffle(filenames)
        self.rand_images = [dst for dst in self.load_images(filenames,sample_size)]



    def load_filenames_from_dir(self,filedir):
        for (path, dirs, files) in os.walk(filedir):
                for f in files:
                    fabs = os.path.abspath(os.path.join(path, f))
                    image_type = imghdr.what(fabs)
                    if not image_type:
                        continue
                    if image_type == 'gif':
                        continue
                    yield fabs

    def load_images(self,filenames,sample_size):
        count = 0
        for l in filenames:
            image_type = imghdr.what(l.rstrip())
            if image_type == 'gif':
                print('{}: gif'.format(l))
                continue
            dst = cv2.imread(l.rstrip())
            yd,xd = dst.shape[:2]
            dratio = float(yd)/float(xd)
            #discard images with extreme height to width ratio
            if dratio > 5.0 or dratio < 0.2:
                print('{}: bad ratio {}'.format(l, dratio))
                continue    
            yield dst
            count+=1
            if count == sample_size:
                break

    def generate_composites(self,src, poly,
                            randomize_loc=False, randomize_scale=False,
                            min_ratio=0.2, scale_lower=0.7, scale_upper=1.3):
        if isinstance(src, basestring):
            src = cv2.imread(src)
        ys, xs = src.shape[:2]

        src_mask = src.copy()
        src_mask = cv2.cvtColor(src_mask, cv2.COLOR_BGR2GRAY)
        src_mask.fill(0)
        cv2.fillPoly(src_mask, [poly], 255)

        # bounding box around polygon
        y1, x1 = np.min(np.nonzero(src_mask), axis=1)
        y2, x2 = np.max(np.nonzero(src_mask), axis=1)

        x_len = x2 - x1
        y_len = y2 - y1

        # crop mask to region of interest and create inverse mask
        mask = src_mask[y1:y2, x1:x2]
        mask_inv = cv2.bitwise_not(mask)

        # cut out the region of interest from the source image
        src1_cut = src[y1:y2, x1:x2]
        # cut out object
        img2_fg = cv2.bitwise_and(src1_cut, src1_cut, mask=mask)

        for dst in self.rand_images:
            yd, xd = dst.shape[:2]
            
            vscale = np.sqrt((xs * ys) / (xd * yd))#difference in image size

            if randomize_scale:
                vscale = np.min((vscale * 1.0/scale_lower, np.sqrt((x_len * y_len) / (xd * yd * 0.20))))
                vscale = np.random.uniform(vscale, vscale * 1.0/scale_upper) 
            else:
                # minimum scale factor, so that object is not less than x percent
                # of image volume (we don't want it to be too small)
                # In this case we set the max scale to a 20% ratio
                 vscale = np.min((vscale, np.sqrt((x_len * y_len) / (xd * yd * min_ratio))))
                    
            # dst image needs to be big enough for object
            scalefactor = np.max((y_len / yd, x_len / xd, vscale))
            dst_r = cv2.resize(dst, dsize=(int(np.ceil(scalefactor * xd)),
                                           int(np.ceil(scalefactor * yd))))
            yd, xd = dst_r.shape[:2]  
 
            ### randomize location
            xtmp = xd - x_len
            ytmp = yd - y_len
            if randomize_loc:
                y_off = 0
                x_off = 0
                if ytmp > 0:
                    y_off = np.random.randint(ytmp)
                if xtmp > 0:
                    x_off = np.random.randint(xtmp)
            else:
                y_off = y1
                x_off = x1
                if y_off > ytmp:
                    y_off = ytmp
                if x_off > xtmp:
                    x_off = xtmp
            y1t, y2t = y_off, y_off + y_len
            x1t, x2t = x_off, x_off + x_len

            # create region of interest
            roi = dst_r[y1t:y2t, x1t:x2t]

            img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Put object in ROI and modify the main image
            dst_r[y1t:y2t, x1t:x2t] = cv2.add(img1_bg, img2_fg)

            # NOTE: image is in cv2 format (0-255, BGR)
            # caffe io format is (0-1, RGB)
            # Convert to (0-1, RGB): (dst_r / 255.0)[:, :, (2, 1, 0)]
            #we don't do the conversion because we disable that conversion in the caffe transformer.
            yield dst_r

    def generate_complement_composite(self,src,poly):
        if isinstance(src, basestring):
            src = cv2.imread(src)
        #create another mask that will be used to get the negative composite
        cvxpoly = cv2.convexHull(poly)
        cvxmask = src.copy()
        cvxmask = cv2.cvtColor(src_mask,cv2.COLOR_BGR2GRAY)
        cvxmask.fill(0)

        #bounding box around polygon
        y1,x1 = np.min(np.nonzero(cvxmask),axis=1)
        y2,x2 = np.max(np.nonzero(cvxmask),axis=1)

        cv2.fillPoly(cvxmask, [cvxpoly], 255)
        center = ((x1+x2)/2,(y1+y2)/2)
        try:
            dst_neg = cv2.seamlessClone(dst,src,cvxmask , center, cv2.NORMAL_CLONE)
        except:
            dst = cv2.resize(dst,src.shape[:-1][::-1])
            dst_neg = cv2.seamlessClone(dst,src,cvxmask , center, cv2.NORMAL_CLONE)
        yield dst_neg

        
