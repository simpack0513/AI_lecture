import numpy as np

class Node:
	# 노드 초기화 - learning_rate, x0, 초기화 후 weight는 표준정규분포를 따르는 랜덤값
	# n은 입력값의 개수이다. 현재로는 2로 사용할 예정
	def __init__(self, n):
		self.n = n
		self.errors = []
		self.learning_rate = 0.05
		self.x0 = 1
		self.weights = np.random.normal(size=n+1)

	def learning(self, x, y):
		# train 데이터 수 만큼 반복
		for i in range(len(y)):
			# output 계산
			output = self.calculate_output(x, i)

			# w0 먼저 계산
			self.weights[0] += self.learning_rate * (y[i] - output) * 1 * self.x0
			# w1, w2, ... wn 차례로 계산
			for j in range(0, self.n):
				self.weights[j+1] += self.learning_rate * (y[i] - output) * 1 * x[j][i]

		# 정확도 계산 (0~1)
		count = 0
		for i in range(len(y)):
			new_output = self.calculate_output(x, i)
			if new_output == y[i]:
				count += 1
		error = count / len(y)
		self.errors.append(1 - error)


	def calculate_output(self, x, i):
		total = 0
		# total += w0x0
		total += self.x0 * self.weights[0]

		# total += w_i * x_i
		for j in range(0, self.n):
			total += x[j][i] * self.weights[j+1]

		# activate function : step
		if total > 0:
			return 1
		else:
			return 0
