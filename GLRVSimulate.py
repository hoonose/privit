from __future__ import print_function
import functions as f
import math
import testers as t

alpha = 0.01
epsilon = 0.1
largeFrac = 1.0/200
alphaFrac = 1.0/3
iterations = 100
alphaList = [0.1, 0.05, 0.01]
nList = [100, 500, 2000, 5000, 10000]

for alpha in alphaList:
	output_filename = "outputs/GLRV_alpha%.4f_epsilon%.2f.txt"%(alpha, epsilon)
	target = open(output_filename, 'a+')
	print("filename: %s\n"%(output_filename))
	print("alpha: %f, epsilon: %f\n"%(alpha, epsilon), file=target)
	print("alpha: %f, epsilon: %f\n"%(alpha, epsilon))

	m = 100
	for n in nList:
		q = f.getUniformDist(n)
		p = f.getPaninski(n, alpha)

		sTester = t.GLRV_tester(epsilon)
		sTester.set_q(q)
		
		m = f.getMinimumSampleSize(sTester, p, m, iterations, 2.0/3)
		print("n: %d, m: %d"%(n, m), file=target)
		target.flush()
		print("n: %d, m: %d, ratio: %.3f"%(n, m, 1.0*m/n))

	target.close()
			
