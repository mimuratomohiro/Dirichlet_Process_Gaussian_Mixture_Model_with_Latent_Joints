# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import scipy.stats  as ss
import sklearn.cluster
import random
import math
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn                 import mixture



class DPGMM():
	"""docstring for DPGMM_LJ"""
	def __init__(self, data, alpha, components):
		self.data         =  data

		self.num_iter     =  100
		self.MIN_VALUE    =  0.0000001
		self.MAX_VALUE    =  0x1000000
		self.components   =  components
		
		self.alpha        =  alpha
		self.V            =  np.eye(data.shape[1])#np.cov(data,rowvar=0) /(10)
		self.v0           =  data.shape[1]+1
		self.m0           =  np.mean(data, axis=0)
		self.k0           =  0.1

		self.pi           =  self.stick_breaking(self.alpha,self.components)
		self.qi           =  np.zeros((components,data.shape[1]))
		self.zi           =  np.random.multinomial(1,self.pi,data.shape[0])
		self.mu           =  sklearn.cluster.KMeans(n_clusters=components, random_state=random.randint(1,100)).fit(data).cluster_centers_
		self.sigma        =  [np.eye(data.shape[1])for n in xrange(self.components)]
		

	def stick_breaking(self,alpha, k):
		betas = np.random.beta(1, alpha, k)
		remaining_pieces = np.append(1, np.cumprod(1 - betas[:-1]))
		p = betas * remaining_pieces
		return p/p.sum()

	def multi_normal_product(self,mu1,mu2,sigma1,sigma2):
		sigma = np.linalg.inv(np.linalg.inv(sigma1)+np.linalg.inv(sigma2))
		mu    = sigma.dot(np.linalg.inv(sigma1).dot(mu1)+np.linalg.inv(sigma2).dot(mu2))
		ans   = ss.multivariate_normal.rvs(mu,sigma)
		return ans


	def fit(self):
		for gibbs in xrange(self.num_iter):
			print gibbs
			self.gibbs_sampling()
		
	def gibbs_sampling(self):
		self.zi_estimation(self.data,self.components,self.mu,self.sigma,self.pi)
		self.pi_estimation(self.zi,self.alpha)
		self.mu_sigma_estimation(self.data,self.zi,self.components)
			
	def zi_estimation(self,data,components,mu,sigma,pi):
		self.new_pi = np.zeros((data.shape[0],components))
		for i in xrange(components):
			self.new_pi[:,i] = ss.multivariate_normal.pdf(data,mu[i],sigma[i])*pi[i]

		
		for i in xrange(data.shape[0]):
			if self.new_pi.sum(1)[i] != 0:
				self.pi_sub = (self.new_pi[i].T/self.new_pi.sum(1)[i]).T
				self.zi[i]  = np.random.multinomial(1,self.pi_sub)
			
	def pi_estimation(self,zi,alpha):
		new_hist_zi = zi.sum(0)
		self.pi     = np.random.dirichlet(new_hist_zi+alpha)

	def mu_sigma_estimation(self,data,zi,components):
		hist           = zi.sum(0)
		self.number    = zi.sum(0)
		self.mean      = np.zeros((components,data.shape[1]))
		self.Vn        = [np.linalg.inv(self.V) for i in xrange(components)]
		self.mn        = [self.m0 * self.k0     for i in xrange(components)]
		self.kn        = self.k0 + zi.sum(0) 
		self.vn        = self.v0 + zi.sum(0) 
		for i in xrange(components):
			if hist[i] != 0:
				self.sub_data   =  data * np.array([zi[:,i] for j in xrange(data.shape[1])]).T
				self.mean[i]    = (self.mean[i] + self.sub_data.sum(0))/float(self.number[i])
				self.mn[i]      = (self.mn[i]   + self.sub_data.sum(0))/self.kn[i]
				self.var        = np.array([self.mean[i] for t in range(data.shape[0])]) * np.array([zi[:,i] for j in xrange(data.shape[1])]).T
				self.Vn[i]     += (self.sub_data - self.var).T.dot(self.sub_data - self.var) + \
									self.k0 * self.number[i] / self.kn[i] * (self.mean[i] -self.m0)[:, np.newaxis].dot((self.mean[i] -self.m0)[np.newaxis,:])

				self.sigma[i]   =  ss.invwishart.rvs(self.vn[i],self.Vn[i])
				self.mu[i]      =  ss.multivariate_normal.rvs(self.mn[i], np.linalg.inv(self.Vn[i]) / self.kn[i])
			else:
				self.sigma[i]   =  ss.invwishart.rvs(self.v0,self.V)
				self.mu[i]      =  ss.multivariate_normal.rvs(self.m0, np.linalg.inv(self.V) / self.k0)

	def likehood(self,data):
		log_likehood = 0

		return like_hood
if __name__ == '__main__':
	
	n_samples = 50
	dimetion  = 15
	X = np.r_[np.random.randn(n_samples,dimetion ),np.random.randn(n_samples, dimetion )+np.ones(dimetion )*10]
	X = np.r_[X,  np.random.randn(n_samples, dimetion )+np.ones(dimetion )*30]

	dpgmm     = DPGMM(X,0.125,10)
	dpgmm.fit()
	label     = np.r_[np.zeros(n_samples),np.ones(n_samples)] 
	label     = np.r_[label,np.ones(n_samples)*2]
	print adjusted_rand_score(label,dpgmm.zi.dot(range(dpgmm.components)))


