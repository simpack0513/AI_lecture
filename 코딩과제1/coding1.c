#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

struct node
{
	int *input;
	double *weight;
} typedef Node;

double gaussianRandom(void);
void set_random_weight(Node *node, int n);
int check_AND_gate(Node *node, int n);

int main() {
	int n;
	int count = 0;
	srand(time(NULL));

	// 노드 동적할당
	Node *node = (Node *)malloc(sizeof(node));

	//입력값 n의 개수 Input
	printf("입력값의 개수(n)를 입력하세요 : ");
	scanf("%d", &n);

	//Node 원소 초기화
	node->input = (int *)malloc(sizeof(int)*n);
	node->weight = (double *)malloc(sizeof(double)*n);

	do {
		count += 1;
		//모든 weight를 랜덤값으로 초기화
		set_random_weight(node, n);

		//매 랜덤 weight로 AND 게이트 값을 넣어서 검증
	} while (check_AND_gate(node ,n) != 1);


	//결과 출력
	printf("Count : %d\n", count);
	printf("weight => ");
	for(int i=0; i<n; i++) {
		printf("%f ", node->weight[i]);
	}

}

//-1~1의 정규분포를 따르는 랜덤 실수 반환
double gaussianRandom() {
  double v1, v2, s;

  do {
    v1 =  2*((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
    v2 =  2*((double) rand() / RAND_MAX) - 1;      // -1.0 ~ 1.0 까지의 값
    s = v1 * v1 + v2 * v2;
  } while (s >= 1 || s == 0);
  s = sqrt( (-2 * log(s)) / s );
  return v1 * s;
}

//모든 weight를 랜덤값으로 초기화
void set_random_weight(Node *node, int n) {
	for(int i=0; i<n; i++) {
		node->weight[i] = 2*((double) rand() / RAND_MAX) - 1;
	}
}

//입력값의 모든 경우의 수를 계산하여 AND_gate와 동일한 값인지 확인
int check_AND_gate(Node *node, int n) {
	//Activate 함수의 역치값
	int threshold = 1;
	//입력값의 모든 경우의 수
	for(int i=0; i<pow(2, n); i++) {
		double total = 0;
		for(int j=0; j<n; j++) {
			node->input[j] = (i >> j) & 1;
		}
		for(int j=0; j<n; j++) {
			total += (node->input[j] * node->weight[j]);
		}
		if (i != pow(2, n) - 1) {
			if (total > threshold)
				return 0;
		}
		else {
			if (total <= threshold)
				return 0;
		}
	}

	return 1;
}
