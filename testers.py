import math
import numpy as np
import random
from scipy.stats import chi2
from collections import Counter

class PrivIT_Laplace_tester:

	def __init__(self, alpha=0.1, epsilon=0.1, delta=0.0, correctProb=2.0/3, C1=0.25, C2=0.5, B1=4.0, B2=1.0, F=1.0, w=0.5, iterations=1000):
		self.alpha = alpha
		self.epsilon = epsilon
		self.delta = delta
		self.correctProb = correctProb
		self.C1 = C1
		self.C2 = C2
		self.B1 = B1
		self.B2 = B2
		self.F = F
		self.w = w
		self.iterations = iterations
		print(2.0*self.delta)
		print(self.epsilon)
		if 2.0*self.delta >= self.epsilon:
			print("Large delta")
			self.s1 = self.C2*self.epsilon + 2.0*self.w*self.delta
			self.s2 = 2.0*(1.0 - self.w)*self.delta
		else:
			print("Small delta")
			self.s1 = self.w*self.C2*self.epsilon + 2.0*self.delta
			self.s2 = (1.0 - self.w)*self.epsilon
		self.m = 1

	def set_q(self, q):
		self.q = q
		self.n = len(q)
		self.A = np.where(q > 1.0*self.C1*self.alpha/len(q))[0]
		self.T = -1.0/(self.s1)*math.log(1 - math.pow(1 - self.C2, 1.0/len(self.A))) # Calculate Laplace threshold to fail with probability C2
		self.set_sensitivity()		
		self.set_threshold()

	def set_m(self, new_m):
		if new_m != self.m:
			self.m = new_m
			self.set_sensitivity()
			self.set_threshold()

	def set_sensitivity(self):
		q_min = 1.0
		for q_i in self.q:
			if q_i < q_min and q_i > 1.0*self.C1*self.alpha/self.n:
				q_min = q_i
		self.sensitivity = (2.0/(self.m*q_min))*((1.0 + self.F)*self.T + max(self.B1*math.sqrt(1.0*self.m*q_min*math.log(len(self.A))), self.B2*math.log(len(self.A))) + 1)

	def set_threshold(self):
		num_true = 0
		num_false = 0
		Z_vals = []
		expected = [self.m*q_i for q_i in self.q]
		for j in range(self.iterations):
			N = self.getSamples(self.q, self.m)
			flag = False
			L = np.random.laplace(0, scale=1.0/self.s1, size=self.n)
			for i in np.nditer(self.A):
				if abs(L[i]) > self.T:
					flag = True
			if flag:
				if random.uniform(0, 1) > 0.5:
					num_true += 1
				else:
					num_false += 1
				continue
			for i in np.nditer(self.A):
				if abs(N[i] + L[i] - expected[i]) > self.F*self.T + max(self.B1*math.sqrt(expected[i]*math.log(len(self.A))), self.B2*math.log(len(self.A))):
					flag = True
			if flag:
				num_false += 1
				continue
			Z = 0 # Calculate statistic
			for i in np.nditer(self.A):
				Z += 1.0*(math.pow(N[i] - self.m*self.q[i], 2) - N[i])/(self.m*self.q[i])
			Z = Z + np.random.laplace(0, self.sensitivity/self.s2)
			Z_vals += [Z]
		Z_vals.sort()
		#print(num_false - num_true)
		Z_index = int(min(max(math.ceil(self.correctProb*self.iterations) - num_true, 0), len(Z_vals) - 1))
		self.threshold = Z_vals[Z_index]

	@staticmethod
	def getSamples(p, m):
		n = len(p)
		samples = np.random.choice(n, size=m, p=p)
		counts = Counter(samples)
		return [counts[i] for i in range(n)]


	def get_constants(self):
		return {"alpha": self.alpha, "epsilon": self.epsilon, "delta": self.delta, "C1": self.C1, "C2": self.C2, "B1": self.B1, "B2": self.B2, "F": self.F}
				

	def test(self, N):
		m = sum(N)
		self.set_m(m)
		L = np.random.laplace(0, scale=1.0/self.s1, size=self.n)
		expected = [m*q_i for q_i in self.q]
		for i in np.nditer(self.A):
			if abs(L[i]) > self.T:
				return random.uniform(0, 1) > 0.5
		for i in np.nditer(self.A):
			if abs(N[i] + L[i] - expected[i]) > self.F*self.T + max(self.B1*math.sqrt(expected[i]*math.log(len(self.A))), self.B2*math.log(len(self.A))):
				return False
		Z = 0 # Calculate statistic
		for i in np.nditer(self.A):
			Z += 1.0*(math.pow(N[i] - m*self.q[i], 2) - N[i])/(m*self.q[i])
		Z = Z + np.random.laplace(0, self.sensitivity/self.s2)
		return Z < self.threshold

class PrivIT_Flip_tester:

	def __init__(self, alpha=0.1, epsilon=0.1, delta=0.0, C1=0.25, C2=0.5, B1=4.0, B2=1.0, F=1.0, S=1.0, w=0.5):
		self.alpha = alpha
		self.epsilon = epsilon
		self.delta = delta
		self.C1 = C1
		self.C2 = C2
		self.B1 = B1
		self.B2 = B2
		self.F = F
		self.S = S
		self.w = w

	def set_q(self, q):
		self.q = q
		self.A = np.where(q > 1.0*self.C1*self.alpha/len(q))[0]
		self.T = -1.0/(self.w*(self.C2*self.epsilon+2.0*self.delta))*math.log(1 - math.pow(1 - self.C2, 1.0/len(self.A))) # Calculate Laplace threshold to fail with probability C2

	def get_constants(self):
		return {"alpha": self.alpha, "epsilon": self.epsilon, "C1": self.C1, "C2": self.C2, "B1": self.B1, "B2": self.B2, "F": self.F, "S": self.S}

	def test(self, N):
		m = sum(N)
		n = len(self.q)
		L = np.random.laplace(0, scale=1.0/(self.w*(self.C2*self.epsilon+2.0*self.delta)), size=n)
		expected = [m*q_i for q_i in self.q]
		for i in np.nditer(self.A):
			if abs(L[i]) > self.T:
				return random.uniform(0, 1) > 0.5
		for i in np.nditer(self.A):
			if abs(N[i] + L[i] - expected[i]) > self.F*self.T + max(self.B1*math.sqrt(expected[i]*math.log(len(self.A))), self.B2*math.log(len(self.A))):
				return False
		Z = 0 # Calculate statistic
		for i in np.nditer(self.A):
			Z += 1.0*(math.pow(N[i] - m*self.q[i], 2) - N[i])/(m*self.q[i])
		Z = self.S*Z/(m*math.pow(self.alpha, 2)) # Scale statistic for better privacy
		Z += 0.7*(1 - self.S) # Shift statistic
		Z = max(Z, 0)
		Z = min(Z, 1)
		return random.uniform(0, 1) > Z

	def getDPBound(self):
		logm = 1.0
		while self.getPrivacy(int(math.floor(math.exp(logm)))) > (1-self.w)*(self.C2*self.epsilon/2.0 + self.delta):
			logm += 0.01
		return int(math.floor(math.exp(logm)))

	def getPrivacy(self, m):
		n = len(self.q)
		q_min = 1.0
		for q_i in self.q:
			if q_i < q_min and q_i > 1.0*self.C1*self.alpha/n:
				q_min = q_i
		return self.S*2.0/(math.pow(m, 2)*math.pow(self.alpha, 2)*q_min)*((1.0 + self.F)*self.T + max(self.B1*math.sqrt(1.0*m*q_min*math.log(len(self.A))), self.B2*math.log(len(self.A))) + 1)

class KR_tester:

	def __init__(self, rho=0.005):
		self.rho = rho
		self.m = 1

	def set_q(self, q):
		self.n = len(q)
		self.q = q
		self.P = np.identity(self.n) - 1.0/self.n*np.outer(np.ones(self.n), np.ones(self.n))
		self.Sigma = np.diag(self.q) - np.outer(self.q, self.q)
		self.set_Projection()

	def set_m(self, new_m):
		if new_m != self.m:
			self.m = new_m
			self.set_Projection()
			
	def set_Projection(self):
		Sigma_centered = self.Sigma + 1.0/(self.m*2.0*self.rho)*np.identity(self.n)
		Sigma_centered_inv = np.linalg.inv(Sigma_centered)
		self.M = self.P.dot(Sigma_centered_inv).dot(self.P)
		
	def test(self, N):
		m = sum(N)
		self.set_m(m)
		X = N + np.random.normal(0, scale=math.sqrt(1.0/(2.0*self.rho)), size=self.n)
		X_centered = X - self.m*self.q
		T = 1.0/m*X_centered.dot(self.M).dot(X_centered)
		return T < chi2.ppf(2.0/3, self.n - 1)

class GLRV_tester:

	def __init__(self, epsilon=0.1):
		self.epsilon = epsilon

	def set_q(self, q):
		self.q = q

	def test(self, N):
		n = len(self.q)
		m = sum(N)
		err_rate = 1.0/3
		k = int(math.ceil(2.0/err_rate))
		samples = []
		for i in range(k):
			samples.append(self.getPrivateChiStat(self.getSamples(self.q, m)))
		samples.sort()
		threshold = samples[int(math.ceil((k + 1)*(1 - err_rate))) - 1]
		return self.getPrivateChiStat(N) < threshold

	def getPrivateChiStat(self, N):
		n = len(self.q)
		m = sum(N)
		Z = 0
		L = np.random.laplace(0, scale=1.0/self.epsilon, size=n)
		for i in range(n):
			Z += 1.0*math.pow(N[i] + L[i] - m*self.q[i], 2)/(m*self.q[i])
		return Z

	@staticmethod
	def getSamples(p, m):
		n = len(p)
		samples = np.random.choice(n, size=m, p=p)
		counts = Counter(samples)
		return [counts[i] for i in range(n)]


