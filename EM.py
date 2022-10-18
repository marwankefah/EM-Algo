import logging
import sys
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster._kmeans import KMeans

logger = logging.getLogger(__name__)
# logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.INFO)


class EM:
    def __init__(self, img, init_type, n_clusters):
        # todo add dim
        self.orig_data = img
        self.nz_indices = [i for i, x in enumerate(img) if x.any()]
        self.data = img[self.nz_indices]

        self.n_clusters = n_clusters

        self.init_type = init_type
        self.init()
        self.responsibilities = None
        self.log_likelihood = 0
        self.seg_labels = None

    def init(self):
        if self.init_type == 'random':
            self.mean_s, self.cov_s, self.pi_s = self.init_random(self.data, self.n_clusters)
        elif self.init_type == 'kmeans':
            self.mean_s, self.cov_s, self.pi_s = self.init_kmeans(self.data, self.n_clusters)
        else:
            raise Exception("Init Type not defined")

    def init_random(self, data, n_clusters):
        mean_s = np.random.randint(low=min(data), high=max(data), size=(n_clusters, data.shape[1]))
        random_assign = np.random.randint(low=0, high=n_clusters, size=data.shape[0])
        cov_s = [np.cov(data[random_assign == i].T) for i in range(n_clusters)]
        pi_s = np.random.rand(n_clusters)
        return mean_s, cov_s, pi_s

    def init_kmeans(self, data, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0)
        y = kmeans.fit_predict(data)
        mean_s = [np.mean(data[y == i], axis=0) for i in range(n_clusters)]
        cov_s = np.array([np.cov(data[y == i].T) for i in range(n_clusters)])
        ids = set(y)
        pi_s = np.array([np.sum([y == i]) / len(y) for i in ids])
        return mean_s, cov_s, pi_s

    def e_step(self, data, mu_s, cov_s, pi_s, n_clusters):
        posterior_probabilities = np.zeros((data.shape[0], n_clusters))
        for k in range(n_clusters):
            posterior_probabilities[:, k] = pi_s[k] * multivariate_normal.pdf(data, mean=mu_s[k], cov=cov_s[k],
                                                                              allow_singular=True)
        # normalize the posterior probabilities
        posterior_probabilities = posterior_probabilities / np.sum(posterior_probabilities, axis=1).reshape(
            (len(posterior_probabilities), 1))
        return posterior_probabilities

    def m_step(self, img, posterior_probabilities, K):
        mu_s = np.zeros((K, img.shape[1]))
        cov_s = np.zeros((K, img.shape[1], img.shape[1]))
        for k in range(K):
            class_posterior = posterior_probabilities[:, k].reshape(1, len(posterior_probabilities[:, k]))
            mu_s[k] = np.matmul(class_posterior, img) / np.sum(class_posterior)
            class_posterior_norm = class_posterior / np.sum(class_posterior)
            cov_s[k] = (class_posterior_norm[0, :] * np.transpose(img - mu_s[k])) @ (img - mu_s[k])
        pi_s = np.sum(posterior_probabilities, axis=0) / posterior_probabilities.shape[0]
        return mu_s, cov_s, pi_s

    def get_labels(self, responsibilities):
        labels = np.argmax(responsibilities, axis=1)
        return labels

    def get_log_likelihood(self, data, mu_s, cov_s, pi_s, n_clusters):
        posterior_probabilities = np.zeros((data.shape[0], n_clusters))
        for k in range(n_clusters):
            posterior_probabilities[:, k] = pi_s[k] * multivariate_normal.pdf(data, mean=mu_s[k], cov=cov_s[k],
                                                                              allow_singular=True)

        log_likelihood = np.sum(np.log(np.sum(posterior_probabilities, axis=1)))
        return log_likelihood

    def get_segm_mask(self, means, labels, orig_data, nz_indices):
        out_labels = np.zeros(orig_data.shape[0])
        data_mean_replaced = np.array([element[0] for element in means])
        em_img = data_mean_replaced[labels]
        out_labels[nz_indices] = em_img
        # todo replace with dim
        img_recovered = out_labels.reshape(240, 240, 48)
        return img_recovered

    def execute(self, tol, max_iter, visualize=True):
        prev_log_likelihood = 100
        iter_n = 0

        self.responsibilities = self.e_step(self.data, self.mean_s, self.cov_s, self.pi_s, self.n_clusters)
        while (abs(prev_log_likelihood - self.log_likelihood) > tol) and (iter_n <= max_iter):
            prev_log_likelihood = self.log_likelihood

            # get inital segmentation labels based on the random init
            self.seg_labels = self.get_labels(self.responsibilities)
            # E-step
            self.responsibilities = self.e_step(self.data, self.mean_s, self.cov_s, self.pi_s, self.n_clusters)
            # M-Step
            self.mean_s, self.cov_s, self.pi_s = self.m_step(self.data, self.responsibilities, self.n_clusters)

            self.log_likelihood = self.get_log_likelihood(self.data, self.mean_s, self.cov_s, self.pi_s,
                                                          self.n_clusters)

            logger.info("iter_n:{}, log_likelihood = {}".format(iter_n,self.log_likelihood))
            if visualize:
                recover_img = self.get_segm_mask(self.mean_s, self.seg_labels, self.orig_data, self.nz_indices)
                plt.imshow(recover_img[:, :, 24], cmap=pylab.cm.cool)
                plt.title('iter' + str(iter_n))
                plt.show()

            iter_n += 1

        logger.info('Converged at iter:{} with loglikelihood:{}'.format(iter_n, self.log_likelihood))

        return self.get_segm_mask(self.mean_s, self.seg_labels, self.orig_data, self.nz_indices)
