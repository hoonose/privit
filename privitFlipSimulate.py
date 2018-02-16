from __future__ import print_function
import functions as f
import math
import testers as t

correctProb = 2.0/3
alpha = 0.1
epsilon = 0.1
delta = 0.0
largeFrac = 1.0/500
alphaFrac = 1.0/3
iterations = 1000

C1 = 1.0
C2 = 0.6
B1 = 2.5
B2 = 1.0
F = 1.0
S = 1.0
w = 0.5


output_filename = "outputs/PrivIT_Flip_alpha%.2f_epsilon%.2f.txt"%(alpha, epsilon)
target = open(output_filename, 'a+')
print("filename: %s\n"%(output_filename))
print("C1:%.5f, C2:%.2f, alpha:%.2f, epsilon:%.2f, delta: %.2f, B1:%.2f, B2:%.2f, F:%.2f, S:%.2f, w:%.2f"%(C1, C2, alpha, epsilon, delta, B1, B2, F, S, w), file=target)
print("C1:%.5f, C2:%.2f, alpha:%.2f, epsilon:%.2f, delta: %.2f,  B1:%.2f, B2:%.2f, F:%.2f, S:%.2f, w:%.2f"%(C1, C2, alpha, epsilon, delta, B1, B2, F, S, w))

n = 2000
m_same = 15000
m_diff = 15000
while True:
	q = f.getUniformDist(n)
	p = f.getPaninski(n, alpha)

	pTester = t.PrivIT_Flip_tester(alpha, epsilon, delta, C1, C2, B1, B2, F, S, w)
	pTester.set_q(q)
	m_same = f.getMinimumSampleSize(pTester, q, m_same, iterations, 1.0 - correctProb, -1)
	print("-"*20)
	m_diff = f.getMinimumSampleSize(pTester, p, m_diff, iterations, correctProb, 1)
	print("n: %d, m_same: %d, m_different: %d, privacy: %d"%(n, m_same, m_diff, pTester.getDPBound()), file=target)
	print("n: %d, m_same: %d, m_different: %d, privacy: %d"%(n, m_same, m_diff, pTester.getDPBound()))
	n = int(min(n*math.exp(0.5), (math.floor(n/200) + 1)*200))
