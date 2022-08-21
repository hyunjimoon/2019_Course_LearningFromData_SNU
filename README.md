05.2019 log
## Project

Sector별로 다른 action

Energy utility가 타 분야와 다르다고 함

강화학습 이론 관련  
dualing DQN, tunneling Q,  
  
how to set states?  
Bellman optimality equation implies: if the value function has been designed well, it is ok to be greedy (locally greedy action = global optimal sequence)k  
  
on-policy vs off-policy:  
episode를 생성하는 policy와 evaluation, importance policy가 같을 때 on  
  
지속적으로 탐험해야 하므로 optimal policy 학습 불가 -> Policy를 두개로 분리:  
1\. target policy: 학습대상이 되는 policy  
2\. behavior policy: behavior결정하는 policy (episode 생성 policy)  
\* on-policy는 target, behavior policy일치하는 off-policy의 special case  
  
Q: action value function  
V: state value function  
policy:

\[latex\] \\pi : S -> A \[/latex\]

  
optimal policy

플젝설명

DataGenerator.py

- **merge\_DataFrame**(index = dates)
- **make\_features**(start\_date, end\_date, is\_training) - return: s\_open, s\_close, features

simulation.py

decision\_ql.py

class QLearningDecisionPolicy

def \_\_init\_\_(self, actions, input\_dim, model\_dir) : return None

def **select\_action**(self, current\_state, is\_training) : return action

def **update\_q**(self, current\_state, action, reward, next\_state) : return None, 대신 마지막줄에 self.sess.run(self.train\_op, feed\_dict={self.x: current\_state, self.y: action\_q\_vals}) 를 통해 뭔가 모형학습시키는듯

def **save\_model**(self, output\_dir, step):

RL\_Training.py

open\_prices, close\_prices, features를 make\_features함수에서 가져오고

action, policy, budget, num\_stocks, num\_epoch을 정의한 후

run\_simulation함

actions를 \*\*\*로 정의함

RL\_Test.py

def test(policy, initial\_budget, initial\_num\_stocks, open\_prices, close\_prices, features) : return portfolio

open\_prices의 길이(=주식종류수일듯)만큼 loop 돌고,

current\_state를 feature, budget, num\_stocks로 update

action을 policy에 따라 여러 action중 current\_state에 의거해 서냍ㄱ

i번째 주식종류는 stock\_price는 시가 중 i번째

포트폴리오 = budget + num\_stocks \* close\_prices\[-1\]

해당 포트폴리오를 리턴

모의투자

사고 파는걸 액션 높을때 사서 낮을때 판다는 액션

특정 피쳐 만들고 스테이트 액션 리워드 정의!

액션을 산의 십승으로 정의도 가능액션을 산의 십승으로 정의도 가능

tensorflow설명

Session 만들어줘야 실행

자료형 byte라 b 찍힘

Sessmrun().decode로 스트링타입변환

그래프 그리고 실행(흘려줌)

Const exec

구성.실행 단계

초기화

F close 안해도 되듯이

With tf.sessions

피클 아닌 체크포인트 확장자 이용

불러올땐 변수초기화 무필요

그냥 프린트 안되고 sess.run해줘야 찍힘

Mnist.train.batch시 자동 다음 배치 생성

한 배치에 대한 소프트맥스 크로스엔트로피

총 600개 배치

Logit output

Softmax apply ->

Tf.nnsoftmax

모의투자

사고 파는걸 액션 높을때 사서 낮을때 판다는 액션

특정 피쳐 만들고 스테이트 액션 리워드 정의!

시장에 영향 안 미침

개장시간에 시가로 주식거래 가정

오픈프라이스로 사서 열흘동안

주식이 없는데 파는 공매도

사고파는건 오픈가격 다편가는 클로즈거격

712345

20일까지 테스트데이터

가만히 있어도 됨

마지막닐 종가

폴더하나 생상 ckpt저장

전체 흐름 볼 수 있는 코스피가 필요하지 않을까?

전자는 다 박살

매도는 시가, 일자별

Decision qql network

애션은 시뮬레이셔 피와이

데이터제에서 데이터 처리

앞 날들의 종가를 피쳐 삼일치사거나 팔거나 가만히 액션

액션을 산의 십승으로 정의도 가능

쌍으로 묶어서 사고 팔수도

순차적으로 할수도

13 왜 +2?

수중돈 주식 개수

지금 주식 개수와 현자 자산이라

큐벨류 최대 선택 여러가지 탐색 못하니까 익스플로레이션랜덤앗성 확률

> 디시젼 큐엘 파이

Q 확률값 y 는

Select action

감마 강의에선 이타로

68줄

0은 그냥 앞에 하나추아

업데으트

시뮬레이션

여러번 해보겠다

론 기뮬레이션

Marketdata.krx.co.kr

시장전보 종목정보 일자별시세

셀크리온 하나가 없고 두개가 더 들어감

야후에 없고 한국사이트에 없는 날짜

2015-08-14

2017-09-22, 2017-12-20

셀트리온엔 있는데 나머지 두개엔 없다

야후에 샐트리온 이상

한국에서 두 개 날따이 해당하는 아홉가 회사

todo: 수식추가 (원리와 배운점)

learning from data bia BLV

baseline model

learning curve

validation

* * *

**1\. neural network**

이론적 배경

feed-forward network

recurrent network

adam

Learning From Data Project1

Neural Network 활용해 미래 미국달러 환율 예측

exchange rate prediction using time-series data

* * *

**min**

t ~ t+9시점의 미국달러 환율 예측오차 (MAE)  
  
$$MAE(y,\\hat{y}) = \\frac{1}{n\_{samples}}\\sum\_{i=0}^{n\_{samples}-1}|y\_{i}-\\hat{y\_{i}}| = 1$$

- step별로 따라가는 것이 아닌, \[t-1 ~ t-n 데이터\] ---> feature ---> t ~ t+9의 Price(종가) 한번에 예측

* * *

**given**

2010년 1월부터 지금까지의 원자재, 환율 데이터

환율

- \[원 to 달러, 호주달러, 유로, 홍콩달러, 파운드, 엔, 위안\]의 price, open, high, low, change% (시가, 저가, 고가, 종가, 전일대비)

원자재

- 금, 은, 구리, 백금, 브렌트유, WTI유, 가솔린, 천연가스의 price, open, high, low, change%

* * *

**using**

training data ---neural network 모델 ---> test data 환율 예측

제출 결과물

- 사용 feature 목록
- 전처리 방법
- 사용 parameter
- 성능 평과 결과

* * *

**결정사항**

1. data 사용 데이터
2. missing data process 결측치 처리방식
3. 테이블 중 사용 column - 현재는 usd만 이용, 타 open, high, low, 타 환율, 원자재 column사용가능
4. input\_days (window) 이전 데이터 몇개를 볼 것인가 x=windowing\_x\[usd, price, input\_days\] 맞추는 건 10개로 고정, 앞에 몇 일을 이용할건가? input\_days 개수만큼 가진 리스트 반환

코드 구조

Data Generator.py

- get\_data\_path(symbol)
- merge\_data(start\_date, end\_date, symbols\_
- make\_features(start\_date, input\_days)
- windowing\_x(data, input\_days)
- windowing\_y(data, input\_days)

training.py

test.py

**Q&A**

Q) DataGenerator.py 파일의 57, 58줄에 x\[-10\], y\[-10\]이 아닌 x\[-10:\], y\[-10:\]이 되어야 하지 않냐는 질문을 해주셨는데,  
여기서 x와 y는 **데이터 값의 list가 아닌 window의 list**이기 때문에 x\[-10\], y\[-10\]가 맞습니다.  
예를 들어 현재 제가 제공해드린 데이터와 기반 코드를 그대로 실행한다면  
x\[-10\]은 2019-03-08부터 2019-03-13까지 5일간의 USD\_price 데이터,  
y\[-10\]은 2019-03-14부터 2019-03-25까지 10일간의 USD\_price 데이터

scikit-learn외에 추가 라이브러리 이용 시 명기

**test.py 실행 시 반드시 MAE 출력!**

* * *

2\. HMM

결과가 잘 안 나왔는데, 아마 최종코드가 validation set에 대해 실험을 못해서 그랬던 것 같다. tmse가 0.59정도가 나왔다 (사진 유)

3\. SVM

클래스 불균형문제, sampling미리하는 것이 컸고

pred\_proba함수 신기했음

multivoting

호기선배 팀 decision function 변화시킨부분 재밌었음

class imb 샘플해주는 라이브러리(산통조교님)와 직접하는 것의 차이?


## Quiz
ma는 low ban? (x)  
naive는 랜덤워크 모형이다  
diap = 0 은 가격을 옳게 예측한 것이다 (ㅇ)  

table 1위의 숫자들에서 left to right이다 (ㅇ)?  
0.01은 state transition을 나타냄 (x) observation state  
(o)  
  
23  
0-1 normalization (x)  
svm은 binary output (ㅇ)  
svm nn의 결과값은 profitability (x) -> accuracy  

24  
C, overfitting관계  
epsilon과 sv 숫자 줄어든다 (x)  
현재 직전 경기 이외 결과는 사용 x (T)  
  
25  
테스트 경기의 배당률로 사용되었다 (F) 안됨  
svm보다 linear사용시 배닏

20 - 25

fig5는 엠이 2인경우  
가우시안(2)는 컴포넌트 개수 (o)

fig3 trend observable(o)  
fig1에서 c3은 production state (x)  
모든 계층은 equivalent한 hmm으로 변환가능하다 (o)

식3에서 와이티는 continous (x)  
daily return (x) -> weekly return

table2 eff>1 는 돈 따는 것 (x)  
식(2)은 나이브 방식 (x) 식3  
two stage k = 7 (o)

fig3는 hmm (x)

* * *

weekly basis로 주식 판다 (x)

co integration은 2개

환율 간 co integration이 이용되지 않았다 (X)  
jump, dip period는 uniform하다 (X)  
논문에서 제안한 모델은 non stationary하다 (X)  
금과 인플레이션은 코릴레이션이 높다 (x)

periodic retraining을 해서 누적수익 재미 봄 (X)

무슨요일인지가 attribute 중 하나였다 (ㅇ)

말 등급이 없는 경우 결측치 처리되었다 (X)

accuracy가 높다고 precision이 높지는 않다고 했다 (O)

5  
저자들은 stock price 예측할 때도 continuous을 사용할 것을 권장 (x)

본 논문에서 사용된 뉴럴네트워크 아웃풋은 stock price 변화량을 나타낸다 (x)

tech indicator들의 0-1 normalization 사용 (x) -> max min

* * *

6
**techincal indicator로 사용되지 않았다 (X)**

**learning rate이다(X) -> model parameter**

**ema가 내려가면 성능좋아짐 (x)**

L2 penalty 이용

input: 자기 제외 t시점, 변동에 대한,

ema는 hidden layer에

동일 business sector끼리만 연결 (hidden input시)

ad가 ema보다 variance 설명을 잘함

j가 500개

수익률 avg, +-는 majority vote

1000개 nn모형 - bootstrap

문턱보다 높을때만(수익률 이상일때만)

Network모형에 ensemble 로 minute 단위의 high-frequency forecasting

특정 stock 이 다른 여러 stock 들과 관련되어 있어

고차원, dependency dynamic 

- Stable 한 performance 를 내는 ensemble NN framework 를 도입했다.
- 단순히 regular double NN 을 사용했을 때 좋은 결과가 나오지 않음
- linear regression model 을 만든 후 영향력 있는 feature 들을 뽑아낸 후 DNN 에 적용시키는 hierarchical 한 구조로 성능을 개선

사용한 evaluation metric 중 sharpe ratio는 risk efficiency를 정량화한다

absoulte return은 5분간격 time point t+1, buy or short selling 

7 Exchange Rate Prediction Using Hybrid Neural Networks and Trading Indicators

**william 등은 svr의 input으로 사용되었다 (x)**

**som이 temporal data에 주로 적용되었다 (X) -> spatial**

**arma는 time series의 stationary를 가정(T)**

rsom previous activity를 current 전에 사용

local 구성 점선이 ema


s

 Recurrent self-organising map 과 support vector regression 을 혼합한 architecture에exponential moving average 와 같은 trading indicator 들을 GA 를 이용해  input 으로 학습시킨 model 로, foreign exchange rate 를 예측하는 데 있어 GARCH 와 같은 global model보다 뛰어난 성능을 기록했다.

- Exchange rate 가 움직이는 규칙을 잡기 위해 moving average indicator 를 사용하였다.
- Recurrent SOM 으로 input space 를 partition 하고, SVR 을 통해 local data 를 fitting 하는 two-stage architecture 로 성능을 향상시켰다.
- 제시한 model 이 quantitative 한 data 와 qualitative 한 data 를 combining 한 경우에도 효과적이다.

8

**insample이 training error (t)**

**2개 Hidden일때 가장 성능이 좋앋ㅆ다(T)**

**period는 하루에 해당한다 (x) -> weekly**

auto corr 있다, stationary weekly exchange

과거 lag가 input node개수

인풋노드 증가시 rmse감소 (6,12,13,15제외)  
  

9 Rolling and recursive neural network

**table1 성능순서(O)**

**21번째 줄은 neuron count (O)**

**slab는 모두 같은 activation function (X)**

rolling은 최신것을 가중치 높게

각 slab별로 다른 활성화함수 이용

on ward, feed forward network

traing size determined for each of the iterations conducted by the program

input: gold price first differences, lagged by one to four periods

network retraining, one step ahead forecasting, back propagation nn with genetic algorithm

m-1개의 forecasting, 지난 예측으로 매번 weight 재학습

aic로 arima(4,1,2)선택

**가장 좋은 모형은 multi step ahead forecast도 적용**

directional accuracy test applied 

bootstrap

rolling > recursive

decaying avg sign prediction은 good performance의미

Rolling: A greater weight on more recent information

10

**크로모좀 하나는 델타값 해당 (ㅇ)**

**ti\_는 binary(x)**

**threshold값은 주어지는 값(X)**

주식의 turning point 예측

min max normalization (selected variable)

5개의 초기 threshold에서 선택 (plr에 적용시 각기 다른 trading signal 반환)

tournament 방식 이용 (representation, selection)

trading signal은 0, 1로 변환되어야 (bf fed into bpn)

3개로 분류(up, down, steady)

trading signal 정의와 price definition은 직접적인 관련 없음

real world에서 techinical index for current day 알 수 없음 -> 이를 해결 위해 buy sell price for trading signal i은 **다음날 opening price**로 결정

uptrend일때 sell

ga와 multistart approach와 비교

11

LG-Trader: Stock Trading Decision Support Based on Feature Selection by Weighted Localized Generalization Error Model

1. **What are the contributions of the manuscript?**

Stock trading 에서의 주요 machine Learning problem 인 classifier architecture selection 과 feature selection 을 동시에 해결하기 위해, GA 를 이용하여 imbalance data 에 적용될 수 있는 weighted GEM 을 minimize 하는 LG-Trader 를 제시하였다.  

- Stock trading 의 data 가 imbalance 되어 있다는 부분에 주목하여, 새로운 measure 인 Weighted Localized Generalization Error 을 제시하였다.
- Optimal 한 input feature 를 고르는 phase 와 decision classification 을 위한 phase 를 나누어 설계하여 feature selection 과 architecture selection 을 모두 해결하였다.
- GA 를 이용하여 imbalance data 문제와 feature/architecture selection 문제를 함께 해결하였다.

- 하루 전의 data 를 input 으로 사용하여, long term dependency 를 capture 하지 못할 수 있다.
- 사용한 trading rule 이 현실에 비해 많이 단순하여, 실제로 이를 이용해 이익을 낼 수 있을지 확신할 수 없다.
- Outlier detection 에 대한 고려를 하지 않았다.

12 Comparative Study of Stock Trend Prediction Using Time Delay, Recurrent and Probabilistic Neural Networks

False alarm 을 limiting 하는 것을 목표로 하여 TDNN, RNN 그리고 PNN 을 비교하였고, 모두 좋은 결과를 보였다.

- Risk/reward ratio 에 주목하여, 이를 향상시키기 위해 limiting false alarm 을 목표로 삼았다.
- Short term prediction 을 효과적으로 진행하였다.
- 세 network 모두 나름의 장점을 가지고 있음을 보였다.
- Model 이 지나치게 복잡한 구조를 가지고 있어, 응용의 범위가 좁다.
- 세 network 의 비교 대상이 linear classifier 인 것에 의문이 든다.
- 세 network 끼리의 비교가 좀 더 심층적으로 진행됐으면 좋았을 것이다.

13 A Hybrid Neurogenetic Approach for Stock Forecasting 

RNN 에 GA 를 합친 hybrid neurogenetic system 을 이용하여 stock forecasting 을 효과적으로 진행하였다.

- GA 를 사용하여 NN 의 weight 를 효과적으로 최적화하였다.
- 36개의 주식을 13개년 동안 예측하는 wide range test 를 진행하여 설득력을 높였다.
- Linux cluster system 을 이용하여 GA 를 parallelize 해 처리 시간을 대폭 줄였다.
- Input feature 를 선정하는 과정에 대한 타당성이 부족하다.
- 목적함수가 수익의 최대화인데, 이는 경제학적인 측면뿐만 아니라 전략적인 요소도 포함되는, NN 의 예측 범위 밖에 있는 영역이라 생각된다.
- 비교 대상이 buy and hold strategy 로 지나치게 단순하다.

14 Predicting Trend in the Next-Day Market by Hierarchical Hidden Markov Model

Hierarchical HMM 을 이용하여 아주 짧은 기간 동안의 stock market index 를 효과적으로 예측하였다.

- 3개의 level 을 가진 hierarchical HMM 을 이용하여 market trend 를 state 에 반영하는 데 성공했다.
- 기존 연구와 비교하여 높은 수준의 accuracy 를 기록했다.
- Stable 한 sequence 와 profitable opportunity 사이의 trade-off 를 고려해 moving average 를 사용하였따.

- 이전 observation 에 대한 관성 때문에, 지속적인 감소 후에 따라오는 큰 증가와 같은 변화의 시작을 잘 잡지 못하는 문제가 존재한다.
- Input feature 에 대한 분석이 부족하다..
- Historical data pattern 을 장기적으로 분석하는 요소가 필요해 보인다.

15 A new application of hidden markov model in exchange rate forecasting

더 많은 데이터 사용에 따른 예측성능 향상이 아닌가 하는 생각이 든다..
