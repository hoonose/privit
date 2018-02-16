import math
import numpy as np
import random
from scipy.stats import chi2
from collections import Counter
import time

def getUniformDist(n):
	return np.array([1.0/n]*n)

def getPaninski(n, alpha):
	pairs = n/2
	return np.array([1.0/n + 1.0*alpha/(pairs*2), 1.0/n - 1.0*alpha/(pairs*2)]*pairs + [1.0/n]*(n%2))

#Make sure to change floor/ceil for *both*
def getSplitUniform(n, largeFrac, alpha, alphaFrac):
	largeN = int(max(math.floor(largeFrac*n), 2))
	smallN = n - largeN
	smallMass = alpha*alphaFrac
	largeMass = 1 - smallMass
	return np.array([largeMass/largeN]*largeN + [smallMass/smallN]*smallN)

def getSplitPaninski(n, largeFrac, alpha, alphaFrac):
	largeN = int(math.floor(largeFrac*n))
	smallN = n - largeN
	smallMass = alpha*alphaFrac
	largeMass = 1 - smallMass
	largePairs = largeN/2
	return np.array([largeMass/largeN + 1.0*alpha/(largePairs*2), largeMass/largeN - 1.0*alpha/(largePairs*2)]*largePairs + [largeMass/largeN]*(largeN%2) + [smallMass/smallN]*smallN)

def getSplitPaninski2(n, largeFrac, alpha, alphaFrac):
	largeN = int(math.floor(largeFrac*n))
	smallN = n - largeN
	smallMass = alpha*alphaFrac
	largeMass = 1 - smallMass
	largePairs = largeN/2
	return np.array([largeMass/largeN + 1.0*(alpha + smallMass)/(largePairs*2), largeMass/largeN - 1.0*(alpha - smallMass)/(largePairs*2)]*largePairs + [largeMass/largeN]*(largeN%2) + [0]*smallN)

def getEvilUniform(n, p):
	return np.array( [(1.0-p)/(n-1)]*(n-1) + [p])

def getEvilPaninski(n, p, alpha):
	pairs = (n-1)/2
	return np.array([(1.0-p)/(n-1) + 1.0*alpha/(pairs*2),(1.0-p)/(n-1) - 1.0*alpha/(pairs*2)]*pairs + [(1.0-p)/(n-1)]*((n-1)%2) + [p])

def getMonoMode(n, alpha):
	return np.array([1.0/n + 1.0*alpha/2] + [1.0/n - 1.0*alpha/(2*(n - 1))]*(n - 1))

def getSamples(p, m):
	n = len(p)
	samples = np.random.choice(n, size=m, p=p)
	counts = Counter(samples)
	return [counts[i] for i in range(n)]

def getDetectionRate(tester, p, m, iterations):
	detected = 0
	for i in range(iterations):
		if not tester.test(getSamples(p, m)):
			detected += 1
	return 1.0*detected/iterations

def getMinimumSampleSize(tester, p, m_start, iterations, threshold, sign=1.0):
	sign = 1.0*sign/abs(sign)
	logm = max(round(math.log(m_start)*20)/20, 2.0)
	while True:
		dr = getDetectionRate(tester, p, int(math.floor(math.exp(logm))), iterations)
		print("logm:%.3f, dr:%.3f"%(logm, dr))
		if (dr - threshold)*sign > 0:
			return int(math.exp(logm))
		logm += 0.10
