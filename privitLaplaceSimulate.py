from __future__ import print_function
import functions as f
import math
import testers as t

correctProb = 2.0/3
alpha = 0.01
epsilon = 0.1
delta = 0.0
largeFrac = 1.0/200
alphaFrac = 1.0/3
iterations = 200
alphaList = [0.1, 0.05, 0.01]
nList = [100, 500, 2000, 5000, 10000]

# epsilon = math.pow(epsilon,2.0)/2.0 + epsilon*math.sqrt(2.0*math.log(1.0/delta)) #comment out if going for "our guarantees imply theirs"

C1 = 1.0
C2 = 0.5
B1 = 2.5
B2 = 1.0
F = 1.0
w = 0.5

for alpha in alphaList:
	output_filename = "outputs/PrivIT_Laplace_alpha%.4f_epsilon%.2f_C2%.2f.txt"%(alpha, epsilon, C2)
	target = open(output_filename, 'a+')
	print("filename: %s\n"%(output_filename))
	print("C1:%.5f, C2:%.2f, alpha:%.4f, epsilon:%.2f, delta: %.2f, B1:%.2f, B2:%.2f, F:%.2f, w:%.2f"%(C1, C2, alpha, epsilon, delta, B1, B2, F, w), file=target)
	print("C1:%.5f, C2:%.2f, alpha:%.4f, epsilon:%.2f, delta: %.2f,  B1:%.2f, B2:%.2f, F:%.2f, w:%.2f"%(C1, C2, alpha, epsilon, delta, B1, B2, F, w))

	m = 10
	for n in nList:
		q = f.getUniformDist(n)
		p = f.getPaninski(n, alpha)

		pTester = t.PrivIT_Laplace_tester(alpha, epsilon, delta, correctProb, C1, C2, B1, B2, F, w, iterations)
		pTester.set_q(q)
		m = f.getMinimumSampleSize(pTester, p, m, iterations, correctProb, 1)
		print("n: %d, m_same: %d, m_different: %d"%(n, m, m), file=target)
		target.flush()
		print("n: %d, m_same: %d, m_different: %d"%(n, m, m))

	target.close()
