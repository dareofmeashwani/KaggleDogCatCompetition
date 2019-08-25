from PIL import Image, ImageDraw, ImageStat
import numpy as np
try:
    import cv2
except:
    cv2=None


class ImageOps:
    def perspectiveTransform(self, images):
        def getMaskCoord(imshape):
            vertices = np.array([[(0.01 * imshape[1], 0.09 * imshape[0]),
                                  (0.13 * imshape[1], 0.32 * imshape[0]),
                                  (0.15 * imshape[1], 0.32 * imshape[0]),
                                  (0.89 * imshape[1], 0.5 * imshape[0])]], dtype=np.int32)
            return vertices

        def getPerspectiveMatrices(X_img):
            offset = 15
            img_size = (X_img.shape[1], X_img.shape[0])
            src = np.float32(getMaskCoord(X_img.shape))
            dst = np.float32([[offset, img_size[1]], [offset, 0], [img_size[0] - offset, 0],
                              [img_size[0] - offset, img_size[1]]])
            perspective_matrix = cv2.getPerspectiveTransform(src, dst)
            return perspective_matrix

        def transform(image):
            perspective_matrix = getPerspectiveMatrices(image)
            image = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]),
                                        flags=cv2.INTER_LINEAR)
            return image

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return transform(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(transform(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def addNoise(self, images, type='gussian'):
        def change(image):
            type = type.lower()
            if (type == 'blur'):
                return cv2.blur(image, (5, 5))
            elif (type == 'gussian'):
                return cv2.GaussianBlur(image, (5, 5), 0)
            elif (type == 'median'):
                return cv2.medianBlur(image, 5)
            elif (type == 'bilateral'):
                return cv2.bilateralFilter(image, 9, 75, 75)

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def addSaltPepperNoise(self, images, salt_vs_pepper=0.2, amount=0.004):
        def change(X_imgs):
            X_imgs_copy = X_imgs.copy()
            row, col, _ = X_imgs_copy.shape
            num_salt = np.ceil(amount * X_imgs_copy[0].size * salt_vs_pepper)
            num_pepper = np.ceil(amount * X_imgs_copy[0].size * (1.0 - salt_vs_pepper))
            for X_img in X_imgs_copy:
                # Add Salt noise
                coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 1
                # Add Pepper noise
                coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_img.shape]
                X_img[coords[0], coords[1]] = 0
            return X_imgs_copy

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def augmentBrightness(self, images, brightness):
        def change(image, bright):
            image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            image1[:, :, 2] = image1[:, :, 2] * bright
            image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
            return image1

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images, brightness)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i], brightness))
            return self.__returnAsNumpyArray(imgs)

    def rotation(self, images, ang_range):
        def change(image):
            ang_rot = np.random.uniform(ang_range) - ang_range / 2
            rows, cols, ch = image.shape
            Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)
            image = cv2.warpAffine(image, Rot_M, (cols, rows))
            return image

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(change(images[i]))
            return self.__returnAsNumpyArray(imgs)

    def shear(self, images, shear_range):
        def change(image):
            rows, cols, ch = image.shape
            pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
            pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
            pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
            pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
            shear_M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, shear_M, (cols, rows))
            return image

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(change(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def translation(self, images, trans_range):
        def change(image):
            rows, cols, ch = image.shape
            tr_x = trans_range * np.random.uniform() - trans_range / 2
            tr_y = trans_range * np.random.uniform() - trans_range / 2
            Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
            image = cv2.warpAffine(image, Trans_M, (cols, rows))
            return image

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return change(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(change(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def horizontalFlip(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return cv2.flip(images, 0)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], 0)))
            return self.__returnAsNumpyArray(imgs)

    def verticalFlip(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return cv2.flip(images, 1)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], 1)))
            return self.__returnAsNumpyArray(imgs)

    def transposeFlip(self, images):
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return cv2.flip(images, -1)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(cv2.flip(images[i], -1)))
            return self.__returnAsNumpyArray(imgs)

    def changeContrast(self, images, level):
        factor = (259 * (level + 255)) / (255 * (259 - level))

        def contrast(c):
            return 128 + factor * (c - 128)

        def change(img):
            return Image.fromarray(img).point(contrast)

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return contrast(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(change(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def saturation(self, images, ratio=0.5):
        import PIL.ImageEnhance as enhance
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = Image.fromarray(images)
            converter = enhance.Color(images)
            return converter.enhance(ratio)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                converter = enhance.Color(img)
                imgs.append(np.array(converter.enhance(ratio)))
            return self.__returnAsNumpyArray(imgs)

    def convertGrayscale(self, images):
        def convert(image):
            image = Image.fromarray(image).convert('L')
            image = np.asarray(image, dtype="int32")
            return image

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return convert(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(convert(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def halftoneImage(self, images, sample=10, scale=1, percentage=0, angles=[0, 15, 30, 45], style='color',
                      antialias=False):
        class Halftone(object):
            def make(self, img, sample=10, scale=1, percentage=0, angles=[0, 15, 30, 45], style='color',
                     antialias=False):
                img = Image.fromarray(img)
                if style == 'grayscale':
                    angles = angles[:1]
                    gray_im = img.convert('L')
                    dots = self.__halftone(img, gray_im, sample, scale, angles, antialias)
                    new = dots[0]
                else:
                    cmyk = self.__gcr(img, percentage)
                    dots = self.__halftone(img, cmyk, sample, scale, angles, antialias)
                    new = Image.merge('CMYK', dots)
                return new

            def __gcr(self, im, percentage):
                cmyk_im = im.convert('CMYK')
                if not percentage:
                    return cmyk_im
                cmyk_im = cmyk_im.split()
                cmyk = []
                for i in range(4):
                    cmyk.append(cmyk_im[i].load())
                for x in range(im.size[0]):
                    for y in range(im.size[1]):
                        gray = min(cmyk[0][x, y], cmyk[1][x, y], cmyk[2][x, y]) * percentage / 100
                        for i in range(3):
                            cmyk[i][x, y] = cmyk[i][x, y] - gray
                        cmyk[3][x, y] = gray
                return Image.merge('CMYK', cmyk_im)

            def __halftone(self, im, cmyk, sample, scale, angles, antialias):
                antialias_scale = 4

                if antialias is True:
                    scale = scale * antialias_scale

                cmyk = cmyk.split()
                dots = []

                for channel, angle in zip(cmyk, angles):
                    channel = channel.rotate(angle, expand=1)
                    size = channel.size[0] * scale, channel.size[1] * scale
                    half_tone = Image.new('L', size)
                    draw = ImageDraw.Draw(half_tone)
                    for x in range(0, channel.size[0], sample):
                        for y in range(0, channel.size[1], sample):
                            box = channel.crop((x, y, x + sample, y + sample))
                            mean = ImageStat.Stat(box).mean[0]
                            diameter = (mean / 255) ** 0.5
                            box_size = sample * scale
                            draw_diameter = diameter * box_size
                            box_x, box_y = (x * scale), (y * scale)
                            x1 = box_x + ((box_size - draw_diameter) / 2)
                            y1 = box_y + ((box_size - draw_diameter) / 2)
                            x2 = x1 + draw_diameter
                            y2 = y1 + draw_diameter

                            draw.ellipse([(x1, y1), (x2, y2)], fill=255)

                    half_tone = half_tone.rotate(-angle, expand=1)
                    width_half, height_half = half_tone.size
                    xx1 = (width_half - im.size[0] * scale) / 2
                    yy1 = (height_half - im.size[1] * scale) / 2
                    xx2 = xx1 + im.size[0] * scale
                    yy2 = yy1 + im.size[1] * scale

                    half_tone = half_tone.crop((xx1, yy1, xx2, yy2))

                    if antialias is True:
                        w = (xx2 - xx1) / antialias_scale
                        h = (yy2 - yy1) / antialias_scale
                        half_tone = half_tone.resize((w, h), resample=Image.LANCZOS)

                    dots.append(half_tone)
                return dots

        h = Halftone()

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return np.array(h.make(images, sample, scale, percentage, angles, style, antialias))
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(h.make(images[i], sample, scale, percentage, angles, style, antialias)))
            return self.__returnAsNumpyArray(imgs)

    def __rgb_to_hsv(self, rgb):
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select(
            [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    def __hsv_to_rgb(self, hsv):
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __returnAsNumpyArray(self, data):
        try:
            temp = None
            for d in data:
                if temp is None:
                    temp = d
                else:
                    temp = np.vstack((temp,np.array(d)))
            return data
        except:
            return data

    def shiftHue(self, images, hout):
        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            hsv = self.__rgb_to_hsv(images)
            hsv[..., 0] = hout
            return np.array(self.__hsv_to_rgb(hsv))
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                hsv = self.__rgb_to_hsv(images[i])
                hsv[..., 0] = hout
                hsv = np.array(self.__hsv_to_rgb(hsv))
                imgs.append(np.array(hsv))
            return self.__returnAsNumpyArray(imgs)

    def randomErasing(self, images, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        import random, math
        def erase(img):
            for attempt in range(50):
                area = img.shape[0] * img.shape[2]
                target_area = random.uniform(sl, sh) * area
                aspect_ratio = random.uniform(r1, 1 / r1)
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w <= img.shape[0] and h <= img.shape[1]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)
                    if img.shape[0] == 3:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                        img[x1:x1 + h, y1:y1 + w, 1] = mean[1]
                        img[x1:x1 + h, y1:y1 + w, 2] = mean[2]
                    else:
                        img[x1:x1 + h, y1:y1 + w, 0] = mean[0]
                    return img
            return img

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return erase(np.array(images))
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(erase(np.array(images[i]))))
            return self.__returnAsNumpyArray(imgs)

    def cropFromCentre(self, images, width, height):
        def crop(img):
            img = Image.fromarray(img)
            old_width, old_height = img.size
            left = (old_width - width) / 2
            top = (old_height - height) / 2
            right = (old_width + width) / 2
            bottom = (old_height + height) / 2
            img = img.crop((left, top, right, bottom))
            return np.array(img)

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return crop(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(crop(images[i])))
            return self.__returnAsNumpyArray(imgs)

    def resizeImage(self, images, width=80, height=80):

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            images = Image.fromarray(images)
            images = images.resize((width, height), Image.ANTIALIAS)
            return np.array(images)
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                img = Image.fromarray(images[i])
                img = img.resize((width, height), Image.ANTIALIAS)
                imgs.append(np.array(img))
            return self.__returnAsNumpyArray(imgs)

    def resizeImageWithAspectRatio(self, images, size=128, pad_option=0):
        def resize(img, size, pad_option):
            height, width = img.shape[0], img.shape[1]
            if height > width:
                pad_size = height - width
                if pad_size % 2 == 0:
                    img = np.pad(img, ((0, 0), (pad_size // 2, pad_size // 2), (0, 0)), 'constant',
                                 constant_values=pad_option)
                else:
                    img = np.pad(img, ((0, 0), (pad_size // 2, pad_size // 2 + 1), (0, 0)), 'constant',
                                 constant_values=pad_option)
            else:
                pad_size = width - height
                if pad_size % 2 == 0:
                    img = np.pad(img, ((pad_size // 2, pad_size // 2), (0, 0),(0,0)), 'constant',
                                 constant_values=pad_option)
                else:
                    img = np.pad(img, ((pad_size // 2, pad_size // 2 + 1),(0,0), (0, 0)), 'constant',
                                 constant_values=pad_option)
            return self.resizeImage(img, size, size)

        if isinstance(images, np.ndarray) and len(images.shape) == 3:
            return np.array(resize(images, size, pad_option))
        elif isinstance(images, np.ndarray) and len(images.shape) == 4 or isinstance(images, (tuple, list)):
            imgs = []
            for i in range(len(images)):
                imgs.append(np.array(resize(images[i], size, pad_option)))
            return self.__returnAsNumpyArray(imgs)

    def readImage(self, filename):
        if isinstance(filename, str):
            return np.array(Image.open(filename))
        elif isinstance(filename, (tuple, list)):
            imgs = []
            for i in range(len(filename)):
                imgs.append(np.array(Image.open(filename[i])))
            return self.__returnAsNumpyArray(imgs)

    def pca_color_augmenataion(self, data):
        import numpy as np
        def data_aug(img, evecs_mat):
            mu = 0
            sigma = 0.1
            feature_vec = np.matrix(evecs_mat)
            se = np.zeros((3, 1))
            se[0][0] = np.random.normal(mu, sigma) * evals[0]
            se[1][0] = np.random.normal(mu, sigma) * evals[1]
            se[2][0] = np.random.normal(mu, sigma) * evals[2]
            se = np.matrix(se)
            val = feature_vec * se
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        img[i, j, k] = float(img[i, j, k]) + float(val[k])
            return img

        res = data.reshape([-1, 3])
        m = res.mean(axis=0)
        res = res - m
        R = np.cov(res, rowvar=False)
        from numpy import linalg as LA
        evals, evecs = LA.eigh(R)
        idx = np.argsort(evals)[::-1]
        evecs = evecs[:, idx]
        evals = evals[idx]
        evecs = evecs[:, :3]
        # evecs_mat = np.column_stack((evecs))
        m = np.dot(evecs.T, res.T).T
        img = []
        for i in range(len(data)):
            img.append(data_aug(data[i], m))
        return np.array(img)
