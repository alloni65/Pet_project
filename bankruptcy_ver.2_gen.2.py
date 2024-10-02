import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import shap
import lime

#Считываем данные с подготовленных CSV файлов
ds_1 = pd.read_csv(r"C:\Dataset\ds_1.csv", thousands=',')
ds_0 = pd.read_csv(r"C:\Dataset\ds_0.csv", thousands=',')

#Сразу добавляем тезисную метрику
ds_1['bankruptsy'] = ds_1.apply(lambda x: 1, axis = 1)
ds_0['bankruptsy'] = ds_0.apply(lambda x: 0, axis = 1)

#ФУНКЦИИ

#Функция для построения графика распределения метрики
def chastota_graph (df, i):           #Чистые данные сюда закидывать!

    plt.subplots(figsize = (8, 8), layout = 'constrained')

    df[i].hist(bins = 300)

    plt.title(f'Распределение {i}')

    plt.xlabel(f'{i}')

    plt.ylabel('Частота')

    return plt.show()

#Функция фильтрации значений в метриках (балансовые метрики)
def ochistka (df):




    active_list = ['2023, Основные средства , RUB',	'2023, Внеоборотные активы, RUB',	'2023, Запасы, RUB', '2023, Чистые активы, RUB',

                '2023, Дебиторская задолженность, RUB', '2023, Денежные средства и денежные эквиваленты, RUB',	'2023, Оборотные активы, RUB', '2023, Выручка, RUB',

                '2023, Прочие доходы, RUB', '2023, Прочие расходы, RUB']

    passive_list = ['2023, Капитал и резервы, RUB', '2023, Кредиторская задолженность, RUB', '2023, Краткосрочные обязательства, RUB']



#Обработка дыр в балансе

    for i in active_list:

        list_test_a = list(np.where(df[i].isna())[0])

        for j in list_test_a:

            fill_value = ((df[i] / df['2023, Активы  всего, RUB']).mean()) * (df.loc[j, '2023, Активы  всего, RUB'])

            df.at[j,i] = fill_value



    for i in passive_list:

        list_test_p = list(np.where(df[i].isna())[0])

        for j in list_test_p:

            fill_value = ((df[i] / df['2023, Пассивы всего, RUB']).mean()) * (df.loc[j, '2023, Пассивы всего, RUB'])

            df.at[j,i] = fill_value





    df = df.dropna(subset = ['2023, Себестоимость продаж, RUB'])

    df = df.dropna(subset = ['2023, Период оборота запасов, дни'])

    df = df.dropna(subset = ['2023, Коэффициент абсолютной ликвидности, %'])

    df = df.dropna(subset = ['2023, Период оборота основных средств, дни'])

    df['2023, Текущий налог на прибыль, RUB'] = df['2023, Текущий налог на прибыль, RUB'].fillna(df['2023, Прибыль (убыток) до налогообложения , RUB'] * 0.8)

    df['2023, Оборачиваемость кредиторской задолженности, разы'] = df['2023, Оборачиваемость кредиторской задолженности, разы'].fillna(df['2023, Выручка, RUB']/df['2023, Кредиторская задолженность, RUB'])

    df['2023, Оборачиваемость запасов, разы'] = df['2023, Оборачиваемость запасов, разы'].fillna(df['2023, Выручка, RUB']/df['2023, Запасы, RUB'])

    df['2023, Оборачиваемость основных средств, разы'] = df['2023, Оборачиваемость основных средств, разы'].fillna(df['2023, Выручка, RUB']/df['2023, Собственный капитал, RUB'])

    df['2023, Оборачиваемость дебиторской задолженности, разы'] = df['2023, Оборачиваемость дебиторской задолженности, разы'].fillna(df['2023, Выручка, RUB']/df['2023, Дебиторская задолженность, RUB'])

    df = df.dropna()

    return df

#Функция построения гистограммы и ящика с усами
def plot_hist_box (df, features):
    num_plots = len(features)
    plt.figure(figsize=(10, 5 * num_plots))
    
    for i, column in enumerate (features):
        #Гистограмма
        plt.subplot(num_plots, 2, 2*i+1)
        df[column].hist(bins = 25)
        plt.title(f'Гистограмма {column}')
        plt.xlabel(column)
        plt.ylabel('Частота')
        #Ящик с усами
        plt.subplot(num_plots, 2, 2*i+2)
        df.boxplot(column = column)
        plt.title (f'Ящик с усами {column}')
        
    plt.tight_layout()
    plt.show()

#Функция переименования метрик
def rename_index (df):
    df = df.rename(columns={'2023, Рентабельность затрат, %': 'rocs',
                            '2023, Рентабельность капитала (ROE), %': 'roe',
                            '2023, Рентабельность активов (ROA), %': 'roa',
                            '2023, Доля себестоимости как процент от выручки, %': 'ratio_cost_revenue',
                            '2023, Рентабельность продаж, %': 'sales_margin',
                            '2023, Чистая норма прибыли, %': 'ros',
                            '2023, Коэффициент концентрации заемного капитала,%': 'ratio_borrowed_equity',
                            '2023, Коэффициент обеспеченности собственными оборотными средствами, %': 'ratio_availablity_own_equity',
                            '2023, Коэффициент маневренности собственных средств, %': 'ratio_assets_equity',
                            '2023, Коэффициент концентрации собственного капитала (автономии), %': 'ratio_autonomy',
                            '2023, Соотношение чистого долга к капиталу, %':'ratio_net_debt_ratio',
                            '2023, Коэффициент соотношения заемных и собственных средств, %': 'ratio_borrowed_own_funds',
                            '2023, Коэффициент оборачиваемости совокупных активов, %': 'ratio_turnover_assets_total',
                            '2023, Доля рабочего капитала в активах компании, %': 'ratio_equity_assets',
                            '2023, Cooтношение дебиторской задолженности к активам компании, %': 'ratio_acc_recievable_assets',
                            '2023, Период оборота активов, дни': 'assets_turnover_period',
                            '2023, Период оборота основных средств, дни':'fixed_assets_turnover_period',
                            '2023, Оборачиваемость основных средств, разы': 'fixed_assets_turnover'   ,                      
                            '2023, Оборачиваемость дебиторской задолженности, разы':'acc_recievable_turnover',
                            '2023, Период погашения дебиторской задолженности, дни': 'recievable_period_acc_payable',
                            '2023, Оборачиваемость запасов, разы':'inventory_turnover',
                            '2023, Период оборота запасов, дни': 'inventory_turnover_period',
                            '2023, Оборачиваемость кредиторской задолженности, разы':'acc_payable_turnover',
                            '2023, Период погашения кредиторской задолженности, дни': 'repayment_period_acc_payable',
                            '2023, EBIT, RUB': 'ebit',
                            '2023, Чистая прибыль (убыток), RUB': 'net_income',
                            '2023, Текущий налог на прибыль, RUB': 'current_tax',
                            '2023, Прибыль (убыток) до налогообложения , RUB': 'inc_loss_bef_tax',
                            '2023, Прочие расходы, RUB': 'loss_others',
                            "2023, Прочие доходы, RUB": 'income_others',
                            '2023, Прибыль (убыток) от продажи, RUB':'income_loss',
                            '2023, Себестоимость продаж, RUB':'net_cost_sales',
                            '2023, Выручка, RUB': 'revenue',
                            '2023, Совокупный долг, RUB': 'gross_debt',
                            '2023, Собственный капитал, RUB': 'own_equity',
                            '2023, Собственный оборотный капитал, RUB': 'own_operational_equity',
                            '2023, Пассивы всего, RUB': 'equity_total',
                            '2023, Краткосрочные обязательства, RUB': 'short_term_liabilities',
                            '2023, Кредиторская задолженность, RUB': 'accounts_payable',                            
                            '2023, Капитал и резервы, RUB': 'equity',
                            '2023, Активы  всего, RUB': 'assets_total',
                            '2023, Оборотные активы, RUB':'current_assets',
                            '2023, Денежные средства и денежные эквиваленты, RUB': 'cash',
                            '2023, Дебиторская задолженность, RUB':'accounts_receivable',
                            '2023, Чистые активы, RUB': 'net_assets',
                            '2023, Запасы, RUB': 'inventories',
                            '2023, Внеоборотные активы, RUB':'non_current_assets',
                            '2023, Основные средства , RUB': 'fixed_assets',
                            'Вид деятельности/отрасль': 'sphere',
                            'Код налогоплательщика':'tax_id',
                            'Возраст компании, лет': 'company_maturity',
                            'Адрес (место нахождения)': 'adress',
                            'Краткое наименование': 'short_name',
                            'Регистрационный номер': 'reg_number',
                            'Наименование':'company_name',
                            '№': 'id',
                            '2023, Коэффициент текущей ликвидности, %': 'KTL',
                                    '2023, Коэффициент быстрой ликвидности, %': 'KBL',
                                    '2023, Коэффициент абсолютной ликвидности, %':'KAL'})
    return (df)

#Функция категоризации видов деятельности
def category (df):
    trade = df['sphere'].str.contains('Торговля|Оборот|Сделка|Рынок', na=False)
    warehouse = df['sphere'].str.contains('Хранение|Консервация|Сбережение|Услуга|Услуги', na=False)
    rent = df['sphere'].str.contains('Аренда|Лизинг|Рента|Съем', na=False)
    produce = df['sphere'].str.contains('Производство|Промышленность|Изготовление|Фабрикация|Колхоз', na=False)

    #Создаем новый признак "niche"
    df.loc[trade, 'niche'] = 'trade'
    df.loc[warehouse, 'niche'] = 'warehouse'
    df.loc[rent, 'niche'] = 'rent'
    df.loc[produce, 'niche'] = 'produce'
    df['niche'] = df['niche'].fillna('others')
    return df

#ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ДАТАСЕТА DS1

#Фильтруем метрики для датасета
feature_list = ds_1.select_dtypes(include = {'float64', 'int64', 'object'}).columns.to_list()
feature_filter = pd.DataFrame({'feature_name': feature_list})
feature_filter['percent'] = feature_filter['feature_name'].apply(lambda x: ds_0[x].isnull().sum())/len(ds_0)
feature_filter = feature_filter[feature_filter['percent'] < 0.4]
new_feature_list = feature_filter['feature_name']
ds_1 = ds_1[new_feature_list]


#Фильтруем ds_1
pure_ds1 = ochistka(ds_1)

pure_ds1 = rename_index (pure_ds1)
features = ('KTL', 'KBL', 'KAL')
#plot_hist_box(pure_ds1, features)

pure_ds1 = category(pure_ds1)

#ПРЕДОБРАБОТКА ДАННЫХ ДЛЯ ДАТАСЕТА DS0

#Фильтруем метрики для датасета
feature_list = ds_0.select_dtypes(include = {'float64', 'int64', 'object'}).columns.to_list()
ds_0 = ds_0[new_feature_list]

#Фильтруем ds_0
pure_ds0 = ochistka(ds_0)
pure_ds0 = rename_index(pure_ds0)

#Изучив графики распределения метрики "Активы", можем заметить, что компании класса 1 (банкроты), не имеют активов больше чем на 1.6 млрд рублей.
#Так как присутствует значительный дисбаланс между количеством представителей каждого класса, придется резать класс 0, исходя из данных класса 1.

#Фильтруем датасет по метрике "Активы"
pure_ds0 = pure_ds0[pure_ds0['assets_total'] < 1.6*10**9]


#Для успешной работы модели нужно кодировать количественные признаки. Наши признаки, на текущем этапе, нужно кластеризовать. Предлагается ввести 5 категорий.

#Формирование категорий
category (pure_ds0)

#В данных присутствуют аномалии. Коэффициенты ликвидности, в среднем, не должны превышать 10 у.е.. В нашем случае, присутствуют выбросы, вплодь до 6000.
#Так как на данном этапе у нас присутствует серьезный дисбаланс классов, избавимся от нерепрезентативных данных, у которых метрики (важность которых является главной гипотезой исследования) являются явными выбросами.
pure_ds0 = pure_ds0.query('KAL < 1')
pure_ds0 = pure_ds0.query('KAL > 0')
pure_ds0 = pure_ds0.query('KBL < 1.5')
pure_ds0 = pure_ds0.query('KBL > 0.5')
pure_ds0 = pure_ds0.query('KTL > 1')
pure_ds0 = pure_ds0.query('KTL < 3')

#plot_hist_box (pure_ds0, features)
pure_ds0.info()  





#----------------------------------------------------------------------------------------------------------------------------#

#Ограничем количество строк в датасете pure_ds0 для баланса классов.
pure_ds0_sampled = pure_ds0.sample(frac=0.3)

#Соединим датасеты
df = pd.concat([pure_ds0_sampled, pure_ds1])
#df.info()

#Восстановим идентификаторы компаний после объединения данных
df['id'] = range(1,len(df) + 1)

#Определимся с метриками для кодирования и скейла
# columns_list = df.columns.to_list()
# print(columns_list)
columns_list = ['id','company_maturity', 'non_current_assets', 'inventories', 'net_assets',
                 'cash', 'current_assets', 'equity',
                   'own_operational_equity', 'revenue', 'income_others',
                     'loss_others', 'inc_loss_bef_tax', 'current_tax', 'repayment_period_acc_payable',
                        'inventory_turnover', 'fixed_assets_turnover',
                        'ratio_acc_recievable_assets', 'ratio_equity_assets', 'ratio_turnover_assets_total',
                        'ratio_borrowed_own_funds', 'ratio_net_debt_ratio', 'ratio_assets_equity', 'ratio_availablity_own_equity',
                        'rocs', 'ratio_cost_revenue', 'roa', 'roe', 'KTL', 'KBL', 'KAL','bankruptsy', 'niche']
# n = 6
# for i in range(0, len(columns_list), n):
#     sub_columns = columns_list[i:i+n]

#     # Строим корреляционную матрицу для этих колонок
#     corr_matrix = df[sub_columns].corr(method='spearman')

#     # Визуализация матрицы корреляции
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
#     plt.title(f'Корреляционная матрица для метрик {i+1} до {i+n}')
#     plt.show()

#По итогам анализа корреляционных матриц, были удалены некоторые метрики, присутствие которых, вызывало бы мультиколлинеарность.
pre_df = df[columns_list]
dict = {'trade':1,'produce':2,'warehouse':3,'rent':4,'others':5}
pre_df['niche'] = pre_df['niche'].map(dict)
#print(pre_df.info())

#Распределим метрики по гурппам для препроцессинга
feature_quality = pre_df.select_dtypes(include = {'object'}).columns.to_list()
feature_quantity = pre_df.select_dtypes(include = {'int64', 'float64'}).columns.to_list()
feature_quantity.remove('bankruptsy')

#print(pre_df[feature_quality])

X = pre_df.drop(['bankruptsy'], axis = 1)
y = pre_df['bankruptsy']
RANDOM_STATE = 42
TEST_SIZE = 0.25
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
random_state = RANDOM_STATE,
test_size = TEST_SIZE,
stratify = y)

ohe_columns = feature_quality
num_columns = feature_quantity
data_preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_columns)
], remainder = 'passthrough')

pipe_final = Pipeline([
    ('preprocessor', data_preprocessor),
    ('models', DecisionTreeClassifier(random_state = RANDOM_STATE))
])
param_grid = [
    {'models': [DecisionTreeClassifier(random_state = RANDOM_STATE)],
    'models__max_features': range (1, 3),
    'models__max_depth': range (1, 3),
    'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {'models': [KNeighborsClassifier()],
    'models__n_neighbors': range (1, 3),
    'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {'models': [LogisticRegression(random_state = RANDOM_STATE)],
    'models__C': range (1, 5),
    'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    },
    {'models': [SVC(random_state = RANDOM_STATE, probability = True)],
    'models__kernel': ['linear', 'rbf'],
    'preprocessor__num': [StandardScaler(), MinMaxScaler(), 'passthrough']  
    }
]
randomized_search = RandomizedSearchCV(
pipe_final,
param_grid,
cv = 5,
scoring = 'f1',
random_state = RANDOM_STATE,
n_jobs = -1)

randomized_search.fit(X_train, y_train)
print('Лучшая модель и её параметры:\n\n',
      randomized_search.best_estimator_)
print('Качество модели на основе метрики f1:\n\n',
      randomized_search.best_score_)

y_test_pred = randomized_search.predict(X_test)
print(f1_score(y_test, y_test_pred))

# Вытягиваем модель из пайплайна
best_model = randomized_search.best_estimator_.named_steps['models']

# Сохраняем в переменную пайплайн с предобработкой данных
preprocessor = randomized_search.best_estimator_.named_steps['preprocessor']

# Записываем в переменные предобработанные тренировочную и тестовую выборки
X_train_preprocessed = preprocessor.transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Вытягиваем названия признаков для дальнейшего построения датафрейма
num_feature_names = num_columns  # численные признаки

all_feature_names = np.concatenate([num_feature_names])

# Строим датафреймы 
X_train_preprocessed_df = pd.DataFrame(X_train_preprocessed, columns=all_feature_names)
X_test_preprocessed_df = pd.DataFrame(X_test_preprocessed, columns=all_feature_names)

# Семплируем данные 
X_train_preprocessed_smpl = shap.sample(X_train_preprocessed_df, 15, random_state=RANDOM_STATE)
X_test_preprocessed_smpl = shap.sample(X_test_preprocessed_df, 15, random_state=RANDOM_STATE)

# Применяем метод SHAP для оценки важности признаков
explainer = shap.KernelExplainer(best_model.predict_proba, X_train_preprocessed_smpl)
shap_values = explainer.shap_values(X_test_preprocessed_smpl)

# Усредняем значения SHAP по всем образцам (если многоклассовая классификация, усредним по классу)
mean_shap_values = np.mean(np.abs(shap_values[1]), axis=1) 

# Создаем датафрейм с весовыми коэффициентами
feature_importance = pd.DataFrame(list(zip(all_feature_names, mean_shap_values)),
                                  columns=['Feature', 'SHAP Importance'])

# Сортируем по важности в порядке убывания
feature_importance_sorted = feature_importance.sort_values(by='SHAP Importance', ascending=False).head(10)

# Выводим результат (топ-10 метрик по важности)
print(feature_importance_sorted)



