# PRML Chapter01 练习题Exercise

## 1.1

![image-20210711010844880](..\img\Chapter01\e1.1.png)

我们要证明我们可以根据这个式子得到我们的$w$的最优解，其实也就是最小化我们的平方损失函数

将1.1的多项式函数代入1.2的平方损失函数中，然后再对我们的$w$求导，最小化我们的函数，可得
$$
\sum_{n=1}^{N}\left(\sum_{j=0}^{M} w_{j} x_{n}^{j}-t_{n}\right) x_{n}^{i}=0
$$
然后我们再换一下位置就可以得到我们的结果

## 1.2

第二题就是用正则化的损失函数写成上述1.122的形式，其实很简单，我们只需要将我们第一题$A_{ij}$替换成$A_{ij}+\lambda I_{ij}$，也就是对其我们加了一个单位矩阵，就是上面的式子，一样的方法证明，很简单。

## 1.3

1.3是一个简单用了贝叶斯概率的问题

首先求拿到苹果的概率
$$
p(a)=p(a|r)p(r)+p(a|b)p(b)+p(a|g)p(g)\\
=\frac{3}{10}×0.2 + \frac{1}{2}×0.2 + \frac{3}{10}×0.6=0.34
$$
第二个问题是求已知拿到的是橙子，求它来自于绿色盒子的概率，这个我们利用贝叶斯公式
$$
p(g|o)=\frac{p(o|g)p(g)}{p(o)}\\
p(o) =p(o|r)p(r)+p(o|b)p(b)+p(o|g)p(g)=0.36
$$
所以我们可以得到我们的结果$p(g|o)$
$$
p(g|o) = \frac{3}{10} × \frac{0.6}{0.36} = \frac{1}{2}
$$

## 1.5

$$
\begin{aligned}
\mathbb{E}\left[(f(x)-\mathbb{E}[f(x)])^{2}\right] &=\mathbb{E}\left[f(x)^{2}-2 f(x) \mathbb{E}[f(x)]+\mathbb{E}[f(x)]^{2}\right] \\
&=\mathbb{E}\left[f(x)^{2}\right]-2 \mathbb{E}[f(x)] \mathbb{E}[f(x)]+\mathbb{E}[f(x)]^{2} \\
&=\mathbb{E}\left[f(x)^{2}\right]-\mathbb{E}[f(x)]^{2}
\end{aligned}
$$
## 1.6

$$
\operatorname{cov}[x, y]=\mathbb{E}[x y]-\mathbb{E}[x] \mathbb{E}[y]
$$
因为我们知道x和y是独立的，所以$p(x, y)=p(x) p(y)$
$$
\begin{aligned}
\mathbb{E}[x y] &=\sum_{x} \sum_{y} p(x, y) x y \\
&=\sum_{x} p(x) x \sum_{y} p(y) y \\
&=\mathbb{E}[x] \mathbb{E}[y]
\end{aligned}
$$
所以最后$cov[x,y]=0$

> 一道题一道题做的有点麻烦，到后面我就跳过，做一些重点标注的题

## 1.10

因为x和z是独立的，所以$p(x,z)=p(x)p(z)$
$$
\begin{aligned}
E[x+z]&=\int\int(x+z)p(x)p(z)dxdz\\
&=\int xp(x)dx + \int zp(z)dz\\
&=E[x]+E[z]
\end{aligned}
$$
由上述结论可以得
$$
(x+z-E[x+z])^2=(x-E[x])^2+(z-E[z])^2+2(x-E[x])(z-E[z])
$$
并且最后两项会被积分到0，因此
$$
\begin{aligned}
var[x+z] &=\int \int (x+z-E[x+z])^2p(x)p(z)dxdz\\
&=\int (x-E[x])^2p(x)dx+\int (z-E[z])^2p(z)dz\\
&=var[x]+var[z]
\end{aligned}
$$

## 1.15

![image-20210711113320982](..\img\Chapter01\e1.15.png)

由于题目有点长，这里为了方便理解，翻译一下

在这个练习和下⼀个练习中，我们研究多项式函数的独立参数的数量与多项式阶数M以及输⼊空间维度D之间的关系。⾸先，我们写下D维空间多项式的M阶项，形式为
$$
\sum_{i_1=1}^D \sum_{i_2=1}^D \dots \sum_{i_M=1}^D w_{i_1,i_2,\ldots,i_M}x_{i_1}x_{i_2}\ldots x_{i_M}
$$
系数$w_{i_1,i_2,\ldots,i_M}$由$D^M$个元素组成，但是独⽴参数的数量远小于此，因为因子$x_{i_1}x_{i_2}\ldots x_{i_M}$有很多互换对称性。首先证明系数的冗余性可以通过把M阶项写成下面的形式的方法消除
$$
\sum_{i_1=1}^D \sum_{i_2=1}^{i_1} \dots \sum_{i_M=1}^{i_{M-1}} \tilde w_{i_1,i_2,\ldots,i_M}x_{i_1}x_{i_2}\ldots x_{i_M}
$$
使用这个结果证明，M阶项的独立参数的数量$n(D|M)$满足下面的递归关系
$$
n(D,M)=\sum^D_{i=1}n(i,M-1)
$$
接下来，使用归纳法证明下面结果成立
$$
\sum^D_{i=1}\frac{(i+M-2)!}{(i-1)!(M-1)!}=\frac{(D+M-1)!}{(D-1)!M!}
$$
可以这样证明：首先证明$D=1$的情况下，对于任意的M，这个结果成立。证明的过程中会用$0!=1$。然后假设这个结论对于D维成立，证明它对D+1维也成立即可。最后，使用之后的两个结果，以及数学归纳法，证明
$$
n(D,M)=\frac{(D+M-1)!}{(D-1)!M!}
$$
可以这样证明：首先证明这个结果对于$M = 2$且任意的$D \ge 1$成立，这可以通过对比联系1.14的
结果得出。然后使⽤公式$（1.135）$和公式$（1.136）$，证明，如果结果对于$M - 1$阶成⽴，那么
它对于M阶也成成立



证明：

为了得到我们$1.135$式子的结果，我们进行一个推导，独立参数的个数$n(D,M)$可以写成
$$
n(D|M)=\sum_{i_1=1}^D \sum_{i_2=1}^{i_1} \dots \sum_{i_M=1}^{i_{M-1}} 1
$$
一共有M项，并且这也可以写成
$$
n(D|M)=\sum_{i_1=1}^D \{\sum_{i_2=1}^{i_1} \dots \sum_{i_M=1}^{i_{M-1}} 1\}
$$
其中大括号中有$M-1$项，并且为$n(i_1,M-1)$，所以我们就可以写成
$$
n(D,M)=\sum^D_{i=1}n(i,M-1)
$$
这就推导出了我们的第$1.135$的式子

接着我们需要推导$1.136$的式子，我们需要用我们数学归纳法
$$
\sum^D_{i=1}\frac{(i+M-2)!}{(i-1)!(M-1)!}=\frac{(D+M-1)!}{(D-1)!M!}
$$
首先当$D=1$时，左边和右边都为1，这个式子是成立的，这里我们用了$0!=1$

接着利用数学归纳法的思想，我们假设我们的式子在任意的D都成立，然后证明他在D+1的情况也是成立的，那么这个式子就是成立的。
$$
\begin{aligned}
\sum_{i=1}^{D+1} \frac{(i+M-2) !}{(i-1) !(M-1) !} &=\frac{(D+M-1) !}{(D-1) ! M !}+\frac{(D+M-1) !}{D !(M-1) !} \\
&=\frac{(D+M-1) ! D+(D+M-1) ! M}{D ! M !} \\
&=\frac{(D+M) !}{D ! M !}
\end{aligned}
$$
所以这个式子当对$D+1$也是满足的，所以，这个式子成立

最后我们来证明$1.137$的式子，我们也是用数学归纳法来证明，首先证明当$M=2$时成立

当$M=2$时，我们可以得到，这个是成立的
$$
n(D,2)=\frac{D(D+1)}{2}
$$
然后我们假设，对于$M-1$来说，这个式子
$$
n(D, M-1)=\frac{(D+M-2) !}{(D-1) !(M-1) !}
$$
把这个式子带入我们前面的$1.135$的右边，我们可以得到
$$
n(D, M)=\sum_{i=1}^{D} \frac{(i+M-2) !}{(i-1) !(M-1) !}
$$
然后我们再将前面的$1.136$带入，我们就可以得到我们的结果
$$
n(D, M)=\frac{(D+M-1) !}{(D-1) ! M !}
$$
因此，对于所有的多项式都是成立。

## 1.16

1.16和1.15是同一类型的题目，这道题他是想证明所有阶数小于等于$M$阶的所有项的独立参数的总数$N(D,M)$，利用前面的结果，我们来证明
$$
N(D,M)=\sum^M_{m=0}n(D,m)=\frac{(D+M)!}{D!M!}
$$
其中$n(D,m)$是m阶项的独立参数的数量

首先当M=0的时候这个式子是明显成立的，我们假设当$M$的时候式子成立，我们需要证明当$M+1$时也成立
$$
\begin{aligned}
N(D,M+1)&=\sum^{M+1}_{m=0}n(D,m)\\
&=\sum^{M}_{m=0}n(D,m)+n(D,M+1)\\
&=\frac{(D+M)!}{D!M!}+\frac{(D+M) !}{(D-1) !(M+1)!}\\
&=\frac{(D+M)!(M+1)+(D+M)!D}{D!(M+1)!}\\
&=\frac{(D+M+1)!}{D!(M+1)!}
\end{aligned}
$$
所以对于M+1时等式也是成立的

当$M\gg D$，在这之中，我们需要用到一个$Stirling$近似，这个近似关系对于大的n是成立的。
$$
n!\simeq n^ne^{-n}
$$

$$
\begin{aligned}
n(D,M)&\simeq \frac{(D+M)^{D+M}e^{-D-M}}{D!M^Me^{-M}}\\
&= \frac{M^{D+M}e^{-D}}{D!M^M}(1+\frac{D}{M})^{D+M}\\
&\simeq \frac{M^De^{-D}}{D!}(1+\frac{D(D+M)}{M})\\
&\simeq \frac{(1+D)e^{-D}}{D!}M^D\\
\end{align}
$$
当然，当$D \gg M$也是相同的，通过计算得$N(10,3)=286$和$N(100,3)=176,851$

## 1.18

![image-20210711135304986](..\img\Chapter01\e1.18.png)

对于这道题，首先作者给出了一个从笛卡尔坐标到极坐标系的公式，我们可以证明$1.143,1.144$
$$
    S_D = \frac{2\pi^{D/2}}{\Gamma(D/2)}
$$

$$
V_D = \frac{S_D}{D}
$$

```python
D = np.linspace(0.1, 20, 1000)
Sd = 2 * np.pi * D / gamma(D / 2)
Vd = Sd / D
```

```python
fig = plt.figure(figsize=(15, 5))

ax = fig.add_subplot(1, 2, 1)
ax.plot(D, Sd)
ax.set_title("Surface Area for unit sphere in $D$ dimensions")
ax.set_xlabel("$D$")
ax.set_ylabel("$S_D$")
ax.grid(alpha=0.6)

ax = fig.add_subplot(1, 2, 2)
ax.plot(D, Vd)
ax.set_title("Volume of a unit sphere in $D$ dimensions")
ax.set_xlabel("$D$")
ax.set_ylabel("$V_D$")
ax.grid(alpha=0.6)
```

![](..\img\Chapter01\e.1.181.png)

## 1.27

这个题目需要我们证明对于不同q的情况下，我们的$y(x)$的取值，当$q=1$时，$y(x)$取中位数，当$q$趋近于0的时候，我们最小误差为条件众数

## 1.30

计算两个高斯分布的KL散度

从书本前面我可以得到$KL(p||q)$
$$
\mathrm{KL}(p \| q)=-\int p(x) \ln q(x) \mathrm{d} x+\int p(x) \ln p(x) \mathrm{d} x
$$
我们将$p$和$q$的高斯分布带入第一个积分，可以得到
$$
\begin{aligned}
-& \int p(x) \ln q(x) \mathrm{d} x=\int \mathcal{N}\left(x \mid \mu, \sigma^{2}\right) \frac{1}{2}\left(\ln \left(2 \pi s^{2}\right)+\frac{(x-m)^{2}}{s^{2}}\right) \mathrm{d} x \\
&=\frac{1}{2}\left(\ln \left(2 \pi s^{2}\right)+\frac{1}{s^{2}} \int \mathcal{N}\left(x \mid \mu, \sigma^{2}\right)\left(x^{2}-2 x m+m^{2}\right) \mathrm{d} x\right) \\
&=\frac{1}{2}\left(\ln \left(2 \pi s^{2}\right)+\frac{\sigma^{2}+\mu^{2}-2 \mu m+m^{2}}{s^{2}}\right)
\end{aligned}
$$
对于我们的第二个积分，我们可以看出来，这是高斯函数的负微分熵，所以我们最后可以写成
$$
\begin{aligned}
\mathrm{KL}(p \| q) &=\frac{1}{2}\left(\ln \left(2 \pi s^{2}\right)+\frac{\sigma^{2}+\mu^{2}-2 \mu m+m^{2}}{s^{2}}-1-\ln \left(2 \pi \sigma^{2}\right)\right) \\
&=\frac{1}{2}\left(\ln \left(\frac{s^{2}}{\sigma^{2}}\right)+\frac{\sigma^{2}+\mu^{2}-2 \mu m+m^{2}}{s^{2}}-1\right)
\end{aligned}
$$
