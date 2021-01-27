import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
import seaborn as sns
import re
from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor


# 对港口类型Embarked特征的处理
def process_Embark(combined_train_test):
    # 缺失值填充，将列按出现频率由高到低排序，众数即第一行，inplace表示原地修改
    combined_train_test['Embarked'].fillna(
        combined_train_test['Embarked'].mode().iloc[0], inplace=True)
    # 字符->数值，返回元组，第一个为映射后的y,第二个数为原先的x
    combined_train_test['Embarked'] = pd.factorize(
        combined_train_test['Embarked'])[0]
    # 获取Embarked特征的 one-hot 编码，按Embarked_a,Embarked_b...排布
    emb_dummies_df = pd.get_dummies(
        combined_train_test['Embarked'],
        prefix='Embarked')
    # 保留one_hot编码和原Embarked数据
    combined_train_test = pd.concat(
        [combined_train_test, emb_dummies_df], axis=1)


# 对sex特征的处理，同港口特征处理
def process_sex(combined_train_test):
    combined_train_test['Sex'] = pd.factorize(combined_train_test['Sex'])[0]
    sex_dummies_df = pd.get_dummies(
        combined_train_test['Sex'],
        prefix='Sex')
    combined_train_test = pd.concat(
        [combined_train_test, sex_dummies_df], axis=1)


# 对名字特征的处理
def process_name(combined_train_test):
    # 首先从名字中提取各种首称呼
    combined_train_test['Title'] = combined_train_test['Name'].map(
        lambda x: re.compile(", (.*?)\.").findall(x)[0])
    # 将名称称呼进行统一化处理
    title_Dict = {}
    title_Dict.update(dict.fromkeys(
        ['Capt', 'Col', 'Major', 'Dr', 'Rev'],
        'Officer'))
    title_Dict.update(dict.fromkeys(
        ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'],
        'Royalty'))
    title_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    title_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    title_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    title_Dict.update(dict.fromkeys(
        ['Master', 'Jonkheer'],
        'Master'))
    combined_train_test['Title'] = combined_train_test['Title'].map(title_Dict)
    # 使用dummy对不同称呼进行分列
    # 先对Title进行正则化
    combined_train_test['Title'] = pd.factorize(
        combined_train_test['Title'])[0]
    title_dummies_df = pd.get_dummies(
        combined_train_test['Title'],
        prefix=combined_train_test[['Title']].columns[0])
    combined_train_test = pd.concat(
        [combined_train_test, title_dummies_df], axis=1)
    # 新添加名字长度的特征
    combined_train_test['Name_length'] = combined_train_test['Name'].apply(len)


# 对Fare特征处理
def process_fare(combined_train_test):
    # 缺失值按照均值填充
    combined_train_test['Fare'] = combined_train_test[['Fare']].fillna(
        combined_train_test.groupby('Pclass').transform(np.mean))
    # 将团体票的票价分配到每个人头上
    combined_train_test['Group_Ticket'] = combined_train_test['Fare'].groupby(
        by=combined_train_test['Ticket']).transform('count')
    combined_train_test['Fare'] = combined_train_test
    ['Fare']/combined_train_test['Group_Ticket']
    combined_train_test.drop(['Group_Ticket'], axis=1, inplace=True)
    # 使用binning给票价分等级：
    combined_train_test['Fare_bin'] = pd.qcut(combined_train_test['Fare'], 5)
    # 对于五个等级的票价任然可以用dummy编码
    combined_train_test['Fare_bin_id'] = pd.factorize(
        combined_train_test['Fare_bin'])[0]
    fare_bin_dummies_df = pd.get_dummies(
        combined_train_test['Fare_bin_id']
        ).rename(columns=lambda x: 'Fare_' + str(x))
    combined_train_test = pd.concat(
        [combined_train_test, fare_bin_dummies_df], axis=1)
    combined_train_test.drop(['Fare_bin'], axis=1, inplace=True)


# Pclass特征处理
def process_pclass(combined_train_test):
    # 建立PClass Fare Category
    def pclass_fare_category(
                            df, pclass1_mean_fare,
                            pclass2_mean_fare, pclass3_mean_fare):
        if df['Pclass'] == 1:
            if df['Fare'] <= pclass1_mean_fare:
                return 'Pclass1_Low'
            else:
                return 'Pclass1_High'
        elif df['Pclass'] == 2:
            if df['Fare'] <= pclass2_mean_fare:
                return 'Pclass2_Low'
            else:
                return 'Pclass2_High'
        elif df['Pclass'] == 3:
            if df['Fare'] <= pclass3_mean_fare:
                return 'Pclass3_Low'
            else:
                return 'Pclass3_High'
    Pclass1_mean_fare = combined_train_test['Fare'].groupby(
        by=combined_train_test['Pclass']).mean().get([1]).values[0]
    Pclass2_mean_fare = combined_train_test['Fare'].groupby(
        by=combined_train_test['Pclass']).mean().get([2]).values[0]
    Pclass3_mean_fare = combined_train_test['Fare'].groupby(
        by=combined_train_test['Pclass']).mean().get([3]).values[0]
    # 建立Pclass_Fare Category
    combined_train_test['Pclass_Fare_Category'] = combined_train_test.apply(
        pclass_fare_category, args=(
                                    Pclass1_mean_fare, Pclass2_mean_fare,
                                    Pclass3_mean_fare), axis=1)
    pclass_level = preprocessing. LabelEncoder()
    # 给每一项添加标签
    pclass_level.fit(np.array(
                            ['Pclass1_Low', 'Pclass1_High',
                                'Pclass2_Low', 'Pclass2_High',
                                'Pclass3_Low', 'Pclass3_High']))
    # 转换成数值
    combined_train_test['Pclass_Fare_Category'] = pclass_level.transform(
        combined_train_test['Pclass_Fare_Category'])
    # dummy 转换
    pclass_dummies_df = pd.get_dummies(
        combined_train_test['Pclass_Fare_Category']).rename(
            columns=lambda x: 'Pclass_' + str(x))
    combined_train_test = pd.concat(
        [combined_train_test, pclass_dummies_df], axis=1)

    combined_train_test['Pclass'] = pd.factorize(
        combined_train_test['Pclass'])[0]


def process_family_category(combined_train_test):
    #  亲友数量合并为一个新特征 Parch and SibSp
    def family_size_category(family_size):
        if family_size <= 1:
            return 'Single'
        elif family_size <= 4:
            return 'Small_Family'
        else:
            return 'Large_Family'

    combined_train_test['Family_Size'] = combined_train_test
    ['Parch'] + combined_train_test['SibSp'] + 1
    combined_train_test['Family_Size_Category'] = combined_train_test
    ['Family_Size'].map(family_size_category)

    le_family =preprocessing.LabelEncoder()
    le_family.fit(np.array(['Single', 'Small_Family', 'Large_Family']))
    combined_train_test['Family_Size_Category'] = le_family.transform(
        combined_train_test['Family_Size_Category'])

    family_size_dummies_df = pd.get_dummies(
        combined_train_test['Family_Size_Category'],
        prefix=combined_train_test[['Family_Size_Category']].columns[0])
    combined_train_test = pd.concat(
        [combined_train_test, family_size_dummies_df], axis=1)


def process_age(combine_train_test):
    #  Age 因为Age项的缺失值较多，所以不能直接填充age的众数或者平均数。
    missing_age_df = pd.DataFrame(
        combined_train_test[[
            'Age', 'Embarked', 'Sex', 'Title', 'Name_length',
            'Family_Size', 'Family_Size_Category',
            'Fare', 'Fare_bin_id', 'Pclass']])
    missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
    missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]

    #  建立Age的预测模型，可以用多模型预测，然后做模型融合
    def fill_missing_age(missing_age_train, missing_age_test):
        missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
        missing_age_Y_train = missing_age_train['Age']
        missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

        # # model 1  gbm
        # gbm_reg = GradientBoostingRegressor(random_state=42)
        # gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
        # gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
        # gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)


        
        # # model 2 rf
        # rf_reg = RandomForestRegressor()
        # rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
        # rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1, scoring='neg_mean_squared_error')
        # rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)

        # joblib.dump(gbm_reg_grid,'model/gbm_age_predict.json')
        # joblib.dump(gbm_reg_grid,'model/rf_age_predict.json')
        gbm_reg_grid = joblib.load('model/gbm_age_predict.json')
        rf_reg_grid = joblib.load('model/rf_age_predict.json')

        missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(
            missing_age_X_test)
        missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(
            missing_age_X_test)

        #  模型融合
        missing_age_test.loc[:, 'Age'] = np.mean(
            [missing_age_test['Age_GB'], missing_age_test['Age_RF']])
        missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

        return missing_age_test

    combined_train_test.loc[
        (combined_train_test.Age.isnull()),
        'Age'] = fill_missing_age(
            missing_age_train, missing_age_test)  # 用模型融合填补缺失值


# ticket特征处理
def process_ticket(combined_train_test):
    # Ticket处理
    combined_train_test['Ticket_Letter'] = combined_train_test
    ['Ticket'].str.split().str[0]
    combined_train_test['Ticket_Letter'] = combined_train_test
    ['Ticket_Letter'].apply(lambda x: 'U0' if x.isnumeric() else x)
    combined_train_test['Ticket_Letter'] = pd.factorize(
        combined_train_test['Ticket_Letter'])[0]

    # Cabin难以预测又不能直接删除，将其分为有和无两类
    combined_train_test.loc[combined_train_test.Cabin.isnull(), 'Cabin'] = 'U0'
    combined_train_test['Cabin'] = combined_train_test['Cabin'].apply(
        lambda x: 0 if x == 'U0' else 1)


# 特征的相关性分析
def correlation(combined_train_test):
    # 特征相关性分析
    Correlation = pd.DataFrame(
        combined_train_test[
                            ['Embarked', 'Sex', 'Title',
                                'Name_length', 'Family_Size',
                                'Family_Size_Category', 'Fare',
                                'Fare_bin_id', 'Pclass',
                                'Pclass_Fare_Category', 'Age',
                                'Ticket_Letter', 'Cabin']])
    colormap = plt.cm.viridis
    plt.figure(figsize=(14, 12))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(Correlation.astype(float).corr(), linewidths=0.1, vmax=1.0,
                square=True, cmap=colormap, linecolor='white', annot=True)

    g = sns.pairplot(
                    combined_train_test[[
                            u'Survived', u'Pclass',
                            u'Sex', u'Age', u'Fare', u'Embarked',
                            u'Family_Size', u'Title', u'Ticket_Letter']],
                    hue='Survived',
                    palette = 'seismic', size=1.2, diag_kind = 'kde',
                    diag_kws = dict(shade=True),
                    plot_kws=dict(s=10))
    g.set(xticklabels=[])
    plt.show()


if __name__ == '__main__':
    train_data = pd.read_csv('Tatanic/data/train.csv')
    test_data = pd.read_csv('Tatanic/data/test.csv')

    test_data['Survived'] = 0  # 新增加一列
    combined_train_test = train_data.append(test_data)
    PassengerId = test_data['PassengerId']

    # 对港口特征处理
    process_Embark(combined_train_test)

    # 对性别特征处理
    process_sex(combined_train_test)

    #  对名字特征的处理
    process_name(combined_train_test)

    # Fare处理
    process_fare(combined_train_test)

    #  Plass特征处理
    process_pclass(combined_train_test)

    # 家庭大小类型特征处理
    process_family_category(combined_train_test)

    # age特征处理
    process_age(combined_train_test)

    # ticket特征处理
    process_ticket(combined_train_test)

    # 特征的相关性分析
    correlation(combined_train_test)

    # 接下来是输入模型前的一些处理
    # Age和fare正则化
    scale_age_fare = preprocessing.StandardScaler().fit(
        combined_train_test[['Age', 'Fare', 'Name_length']])
    train_data['Age']
    train_data[['Age']]
    combined_train_test[[
        'Age', 'Fare', 'Name_length'
        ]] = scale_age_fare.transform(
            combined_train_test[['Age', 'Fare', 'Name_length']])

    # 丢弃掉无用的特征
    combined_train_test.drop([
        'PassengerId', 'Embarked', 'Sex', 'Name',
        'Title', 'Fare_bin_id', 'Pclass_Fare_Category',
        'Parch', 'SibSp', 'Family_Size_Category', 'Ticket'],
        axis=1, inplace=True)

    # 将训练数据和测试数据分开
    combined_train_test[:891].to_csv('data/train2.csv')
    combined_train_test[891:].to_csv('data/test2.csv')
