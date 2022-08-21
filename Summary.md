
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
