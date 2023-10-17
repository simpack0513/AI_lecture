import numpy as np

class Node:
	# 노드 초기화 - learning_rate, x0, 초기화 후 weight는 표준정규분포를 따르는 랜덤값
	# n은 입력값의 개수이다. 현재로는 2로 사용할 예정
	def __init__(self, n, str):
		self.n = n
		self.errors = []
		self.learning_rate = 0.05
		self.x0 = 1
		self.weights = np.random.normal(size=n+1)
		self.activation_f = str
		self.output = 0

	def learning(self, x, front_d):
		if self.activation_f == "ReLU":
			z_prime = 1 if self.output > 0 else 0
		else:
			z_prime = 1
		# w0 먼저 계산
		self.weights[0] += self.learning_rate * front_d * z_prime * self.x0
		# w1, w2, ... wn 차례로 계산
		for j in range(0, self.n):
			self.weights[j+1] += self.learning_rate * front_d * z_prime * x[j]
		return front_d * z_prime

	def calculate_output(self, x):
		total = 0
		# total += w0x0
		total += self.x0 * self.weights[0]

		# total += w_i * x_i
		for i in range(0, self.n):
			total += x[i] * self.weights[i+1]

		# activate function : ReLU
		if self.activation_f == "ReLU":
			if total > 0:
				self.output = total
			else:
				self.output = 0

		# activate function : step
		if self.activation_f == "step":
			if total > 0:
				self.output = 1
			else:
				self.output = 0

	def calculate_error_rate(self, x, y):
		# 정확도 계산 (0~1)
		count = 0
		for i in range(len(y)):
			new_output = self.calculate_output(x, i)
			if new_output == y[i]:
				count += 1
		error = count / len(y)
		self.errors.append(1 - error)

