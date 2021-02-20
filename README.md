# unlimited_power

언리미티드 빠와

# 공동 작업으로 정리 대기 중!!!

작업자 : 조재윤, 최명진, 표유진

2020 데이콘, 태양광 발전량 예측 AI 경진대회

7일 동안의 데이터를 인풋으로 활용하여, 향후 2일간의 발전량을 예측
(데이터는 각 30분 간격으로 주어짐)

Quantile 예측으로 채점
Quantile = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

## 주된 폴더 EDA와 LGBM(최종제출모델) 폴더

### EDA 폴더 내용

##### 각각의 시계열 자료 별 간단한 추이 확인

![image](https://user-images.githubusercontent.com/76254564/107886250-fcd05080-6f41-11eb-910b-df2abdb88fd8.png)

##### 간단한 상관관계와 회귀분석으로 중요 요인 파악

![image](https://user-images.githubusercontent.com/76254564/107886263-170a2e80-6f42-11eb-95f9-ba2edbcb82a2.png)
![image](https://user-images.githubusercontent.com/76254564/107886277-25f0e100-6f42-11eb-8c58-6873ce18737a.png)

##### 정규성 검정과 다중공선성(VIF) 검정

![image](https://user-images.githubusercontent.com/76254564/107886292-3acd7480-6f42-11eb-8d38-9d4038f1f15e.png)

##### 각 시간대 별로 분위 별 분포 시각화

    for i in times:
      dataset = np.array(data[data.Time == i].TARGET)
      time_percentile = []
      for j in range(1,10):
        k = np.percentile(dataset, j*10)
        time_percentile.append(k)
      print('시각 : ', i)
      print(time_percentile)
      plt.figure(figsize=(5, 5))
      plt.plot(time_percentile)
      plt.show()


![image](https://user-images.githubusercontent.com/76254564/107886337-83852d80-6f42-11eb-8e5f-e8f2a2efe2cf.png)

##### 9개 분위별 예측이므로 9개 결과가 분위에 맞도록 사후 수정작업 진행

![image](https://user-images.githubusercontent.com/76254564/107886353-9e57a200-6f42-11eb-8153-8f5533eb025c.png)

##### 예측값을 다른 8개 예측값, 24시간 예측값과 비교하여 수정

##### 0.4 분위는 0.5 분위보다 작아야 하고, 아침 8시의 발전량은 정오의 발전량보다 작아야 하므로, 예측값 이상치 감지 & 후처리

    def columns_IDA(dataset):
      dataset2 = pd.DataFrame()
      for j in range(0,9):
        new_val = []
        for m in range(1, int(raw_data.shape[0]/48)+1):
          temp = dataset.iloc[48*(m-1):48*m,j].values

          strt = 0
          endd = 0
          k = 0

          for i in range(len(temp)):
            if temp[i] > k and k ==0 and i > 10 and i < 24:
              strt = i
            elif temp[i] < k and temp[i] == 0 and i > 24 and i < 40:
              endd = i
            k = temp[i]
          if strt != 0:
            bf = temp[:strt+1]
            y = temp[strt+1:endd-1]
            af = temp[endd-1:]
            # bf = temp[:strt]
            # y = temp[strt:endd]
            # af = temp[endd:]

            x = np.array([i for i in range(len(y))])
            fit_t = np.polyfit(x,y,5)

            for i in x:
              fit1 = fit_t[0]*x**5 + fit_t[1]*x**4 + fit_t[2]*x**3 + fit_t[3]*x**2 + fit_t[4]*x + fit_t[5]
          # fit1 = fit_t[0]*x**4 + fit_t[1]*x**3 + fit_t[2]*x**2 + fit_t[3]*x + fit_t[4]
          # fit1 = fit_t[0]*x**2 + fit_t[1]*x + fit_t[2]

            revised = np.concatenate((bf, fit1, af), axis=0)
          elif strt == 0:
            revised = temp
          new_val.extend(list(revised))
        k = pd.DataFrame(new_val)
        dataset2 = pd.concat([dataset2, k], axis = 1)
      return dataset2

    def f(x, a,b,c, d):
      b = b**2
      y = 1/3 * b * x**3 - a*b *x**2 + (a**2*b + b*c**2)*x + d
      # y = a*(x-12*a) * x **2 + c
      # y = a*x**2 + b
      return y

    def rows_IDA(dataset):
      revised = pd.DataFrame()
      for i in range(dataset.shape[0]):
        y = dataset.iloc[i,:].values
        x = np.array([i for i in range(1, 10)])

        if list(y).count(0) <= 6 :
          popt, pcov = curve_fit(f, x, y, maxfev = 999990000)
          yfit = f(x, *popt)
          revised = pd.concat([revised, pd.DataFrame(yfit).transpose()], axis = 0)
        else:
          revised = pd.concat([revised, pd.DataFrame(y).transpose()], axis = 0)
        if i % 1000 == 0:
          print(i, '/7776')
      return revised


### LGBM 폴더

##### 최종 예측 모델로 선정, 분위(quantile)별로 1개씩, 총 9개 예측모델 생성

    def train_data(X_train, Y_train, X_test):
        LGBM_models=[]
        LGBM_actual_pred = pd.DataFrame()

        for q in quantiles:
            print(q)
            pred, model = LGBM(q, X_train, Y_train, X_test)
            LGBM_models.append(model)
            LGBM_actual_pred = pd.concat([LGBM_actual_pred,pred],axis=1)

        LGBM_actual_pred.columns=quantiles

        return LGBM_models, LGBM_actual_pred


##### 데이터 전처리, 정규화, scaling 등 진행 / 시계열 데이터 학습형태로 정렬

    def train_to_supervised(train, target, n_in):

        clmns = list(train.columns)

        # 기타 칼럼은 전과 같이 들어갈 것.
        scaled_lst = clmns

        scaled_df = train[scaled_lst]
        target_df = target

        # 미래 몇 번째 항목을 가져올 것인가
        future = [48, 96]

        ### 만약에 스케일링을 하고 싶다면 ###
        # scaled_df 데이터 프레임만 스케일링 하고, 절기랑 TARGET 데이터는 그냥 두면 된다.

        # 스케일링 해도 되고, 안해도 되는 기존에 썻던 변수들 전처리
        cols, names = list(), list()
        n_vars = 1 if type(scaled_df) is list else scaled_df.shape[1]
        n_vars2 = 1 if type(target_df) is list else target_df.shape[1]
        for i in range(n_in, 0, -1):
            cols.append(scaled_df.shift(i))
            names += [('%s(t-%d)' % (j, i)) for j in scaled_df.columns]

        # 48과 96 후의 타겟 데이터 2개 붙이기.
        # forecast sequence (t, t+1, ... t+n)
        for i in future:
            cols.append(target_df.shift(-i))
            if i == 0:
                names += [('TARGET%d(t)' % (j+1)) for j in range(n_vars2)]
            else:
                names += [('TARGET%d(t+%d)' % (j+1, i)) for j in range(n_vars2)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        return agg


##### 시간 변수는 00:30 분과 23:30분이 실제로는 1시간 차이인데, 수치상으로 23시간 차이를 보이기 때문에 sin, cos_time으로 변환

##### 기존의 DNI, DHI 변수를 활용해 태양의 남중고도까지 고려하는 GHI 파생변수 추가

    def HRA(DHI, DNI, season, hour):
      # 위도(latitude) 기준을 일단 임의로 대전으로 설정 (위도 36.19~36.2도)
      latitude = radians(36.2)
      season = int(season)
      # 절기별 대한민국의 경사각
      tilt = radians(경사각[season])

      # 절기별 대한민국 대전의 태양 남중시각
      hra = radians(15*(hour - 남중[season]))

      # 구하려는 알파
      elevation = np.arcsin(np.sin(tilt) * np.sin(latitude) + np.cos(tilt) * np.cos(latitude) * np.cos(hra))

      # 천정각(Zenith Angle)은 90 - 알파
      zenith = radians(90) - elevation

      # GHI는 DHI + DNI * cos(천정각)
      ghi = DHI + DNI *np.cos(zenith)

      return ghi

### Catboost 폴더

##### Categorical 변수에 강한 Catboost 예측 모델 적용.

##### 예측 모델 중 하나로 시도했으나, LGBM보다 성능이 좋지 않아 제외.

    def CBST(q, X_train, X_valid, Y_train, Y_valid, X_test):
        # (a) 모델링
        model = CatBoostRegressor(objective='Quantile:alpha={}'.format(q),
                                  eval_metric = 'Quantile',
                                  n_estimators = 15000,
                                  learning_rate = 0.007,
                                  random_state = 42,
                                  bagging_temperature = 0.7,
                                  max_depth = 16,
                                  max_bin = 63,
                                  task_type = 'GPU'
                                  )
        model.fit(X_train, Y_train, eval_set=[(X_valid, Y_valid)], early_stopping_rounds=500, verbose=500)


            # verbose : eval metric이 이 숫자만큼의 round가 지난 다음 자동으로 출력된다.
            # early_stopping_rounds : 이 숫자가 가기 전까지 validation score가 증가하지 않으면 round를 멈춘다.

        # (b) 예측
        pred = pd.Series(model.predict(X_test).round(2))
        return pred, model
