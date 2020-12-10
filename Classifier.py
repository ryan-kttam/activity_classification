from sklearn.preprocessing import MinMaxScaler
import cv2
import numpy as np
import pandas as pd

class Classifier():

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.trainData, self.trainLabel, self.trainVideoLabel = None, None, None
        self.testData, self.testLabel, self.testVideoLabel = None, None, None

    def get_binary_imgs(self, path, theta, tau):
        # Given: a video path
        # calculate motion history images, eliminate images with small values
        # return a list of np.array of images (frames)

        kernel_size = 2
        iter = 1
        video = cv2.VideoCapture(path)
        previous_img = None
        m_t = []
        first_mhi = True
        while (video.isOpened()):
            _, frame = video.read()
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = gray.astype(float)
                if previous_img is not None:
                    difference = abs(gray - previous_img)
                    img_diff = (difference >= theta).astype(np.uint8)
                    tmp = cv2.erode(img_diff, np.ones((kernel_size, kernel_size), dtype=np.uint8),
                                    iterations=iter) * 1.0
                    if first_mhi:
                        tmp *= tau
                        if tmp.sum() > 999:
                            m_t.append(tmp)
                            first_mhi = False
                    else:
                        tmp *= tau
                        zeros = np.zeros(m_t[-1][tmp == 0].shape)
                        temp = m_t[-1][tmp == 0] - 1
                        tmp[tmp == 0] = np.maximum(zeros, temp)
                        if tmp.sum() > 999:
                            m_t.append(tmp)
                previous_img = gray.copy()
            else:
                video.release()
                break
        return m_t

    def calculate_M_Tau(self, x, y=None):
        # Given a folder which contains videos
        # calculate the MHI for each video
        # return a tuple of 2 elements:
        # data: a list of np.arrays
        # video_label: a list of strings containing video's name

        data = []
        video_label = []
        for i in range(len(x)):
            path = x[i]
            theta, tau = 20, 20
            newM_tau = self.get_binary_imgs(path=path, theta=theta, tau=tau)
            data += newM_tau
            video_label += [x[i]] * len(newM_tau)
        return data, video_label

    def calculate_moment(self, M, i, j):
        # Given a list of moments (M), and i,j order, calculate the moment
        # return a list of float

        mij = []
        for idx in range(len(M)):
            x, y = np.where(M[idx] != 0)
            x_i = x ** i
            y_j = y ** j
            moment_ij = (x_i * y_j * M[idx][x, y]).sum()
            mij.append(moment_ij)
        return mij

    def calculate_central_moments(self, M, i, j, xMean, yMean):
        # Given a list of moments (M), i,j orders, and the mean of the moments (x,y)
        # calculate the central moments (scaled and unscaled)
        # return a tuple of 2 elements:
        # a list of unscaled central moments and a list of scaled invariant moments

        mij = []
        sim = []
        for idx in range(len(M)):
            m00 = self.calculate_moment([M[idx]], 0, 0)[0]
            x, y = np.where(M[idx] != 0)
            x_i = (x - xMean[idx]) ** i
            y_j = (y - yMean[idx]) ** j
            moment_ij = (x_i * y_j * M[idx][x, y]).sum()
            scaledInvariantMoment = moment_ij / m00 ** (1 + (i + j) / 2)
            mij.append(moment_ij)
            sim.append(scaledInvariantMoment)
        return np.array(mij), np.array(sim)

    def moments(self, M):
        # data prep for calculating the Hu moments

        x_mean = np.array(self.calculate_moment(M, i=1, j=0)) / np.array(self.calculate_moment(M, i=0, j=0))
        y_mean = np.array(self.calculate_moment(M, i=0, j=1)) / np.array(self.calculate_moment(M, i=0, j=0))

        hu_m20, hu_v20 = self.calculate_central_moments(M, 2, 0, x_mean, y_mean)
        hu_m11, hu_v11 = self.calculate_central_moments(M, 1, 1, x_mean, y_mean)
        hu_m02, hu_v02 = self.calculate_central_moments(M, 0, 2, x_mean, y_mean)
        hu_m30, hu_v30 = self.calculate_central_moments(M, 3, 0, x_mean, y_mean)
        hu_m21, hu_v21 = self.calculate_central_moments(M, 2, 1, x_mean, y_mean)
        hu_m12, hu_v12 = self.calculate_central_moments(M, 1, 2, x_mean, y_mean)
        hu_m03, hu_v03 = self.calculate_central_moments(M, 0, 3, x_mean, y_mean)
        hu_m22, hu_v22 = self.calculate_central_moments(M, 2, 2, x_mean, y_mean)

        unscaled = [hu_m20, hu_m11, hu_m02, hu_m30, hu_m21, hu_m12, hu_m03, hu_m22]
        scaled = [hu_v20, hu_v11, hu_v02, hu_v30, hu_v21, hu_v12, hu_v03, hu_v22]

        return unscaled, scaled

    def calculate_hu_moments(self, moments):
        # Calculate the Hu moments

        hu_m20, hu_m11, hu_m02, hu_m30, hu_m21, hu_m12, hu_m03, hu_m22 = moments

        hu1 = hu_m20 + hu_m02
        hu2 = (hu_m20 - hu_m02) ** 2 + 4 * (hu_m11 ** 2)
        hu3 = (hu_m30 - 3 * hu_m12) ** 2 + (3 * hu_m21 - hu_m03) ** 2
        hu4 = (hu_m30 + hu_m12) ** 2 + (hu_m21 + hu_m03) ** 2
        hu5 = (hu_m30 - 3 * hu_m12) * (hu_m30 + hu_m12) * ((hu_m30 + hu_m12) ** 2 - 3 * (hu_m21 + hu_m03) ** 2) + \
              (3 * hu_m21 - hu_m03) * (hu_m21 + hu_m03) * (3 * (hu_m30 + hu_m12) ** 2 - (hu_m21 + hu_m03) ** 2)
        hu6 = (hu_m20 - hu_m02) * ((hu_m30 + hu_m12) ** 2 - (hu_m21 + hu_m03) ** 2) + \
              4 * hu_m11 * (hu_m30 + hu_m12) * (hu_m21 + hu_m03)
        hu7 = (3 * hu_m21 - hu_m03) * (hu_m30 + hu_m12) * ((hu_m30 + hu_m12) ** 2 - 3 * (hu_m21 + hu_m03) ** 2) - \
              (hu_m30 - 3 * hu_m12) * (hu_m21 + hu_m03) * (3 * (hu_m30 + hu_m12) ** 2 - (hu_m21 + hu_m03) ** 2)

        return [hu1, hu2, hu3, hu4, hu5, hu6, hu7]

    def add_training_data(self,x_train, y_train):
        # add data into the class for later training (if needed)
        # filled in the scaler.

        print('loading data...')
        self.trainData, self.trainVideoLabel = self.calculate_M_Tau(x_train, y_train)
        self.trainLabel = [i[2] for i in pd.Series(self.trainVideoLabel).str.split('_')]

        print('Calculating Moments...')
        train_m = self.moments(self.trainData)
        print('Calculating Hu Moments...')
        unscaled_hu = self.calculate_hu_moments(train_m[0])
        scaled_hu = self.calculate_hu_moments(train_m[1])
        feature_unscaled = unscaled_hu + scaled_hu

        df = pd.DataFrame(np.array(feature_unscaled).T)
        self.scaler.fit(df)
        self.features_train = self.scaler.transform(df)

    def train(self, learner):
        # Given a learner, train the model

        if self.trainData is None:
            print ('Please import training and test data ')

        print('Training the model...')
        learner.fit(self.features_train, self.trainLabel)
        print('Generating Predictions...')
        pred = learner.predict(self.features_train)
        result = pd.DataFrame(pred, columns=['predictions'])
        result['video'] = self.trainVideoLabel
        result['actual'] = self.trainLabel

        result_table = result.groupby('video').agg(lambda x: x.value_counts().index[0])
        # training error
        print ('Training accuracy: ', (result_table.predictions == result_table.actual).sum() / len(result_table))

    def predict(self, learner, x_test, y_test = None):
        # Make predictions based on a given learner and the test set

        print('preprocessing the test data...')
        self.testData, self.testVideoLabel = self.calculate_M_Tau(x_test, y_test)

        self.testLabel = [i[2] for i in pd.Series(self.testVideoLabel).str.split('_')]
        print('Calculating Moments...')
        m = self.moments(self.testData)
        print('Calculating Hu Moments...')
        unscaled_hu = self.calculate_hu_moments(m[0])
        scaled_hu = self.calculate_hu_moments(m[1])
        feature_unscaled = unscaled_hu + scaled_hu

        df = pd.DataFrame(np.array(feature_unscaled).T)

        self.features_test = self.scaler.transform(df)

        print('Generating Predictions...')
        pred = learner.predict(self.features_test)

        result = pd.DataFrame(pred, columns=['predictions'])
        result['video'] = self.testVideoLabel
        if y_test is not None:
            result['actual'] = self.testLabel

        result_table = result.groupby('video').agg(lambda x: x.value_counts().index[0])
        return result_table.predictions