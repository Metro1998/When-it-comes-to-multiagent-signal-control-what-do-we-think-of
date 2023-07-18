from scipy.special import comb

n = 8
k = 2

probability = comb(n, 1) / (2 ** n)
print("每一列有一个1的概率：", probability)