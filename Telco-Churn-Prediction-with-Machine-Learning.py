# Telco Churn Prediction
###############################

# İş Problemi :
##############################

# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

##########################
# Veri Seti Hikayesi
#########################

# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki
# 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.


# CustomerId --> Müşteri İd’si
# Gender --> Cinsiyet
# SeniorCitizen --> Müşterinin yaşlı olup olmadığı(1, 0)
# Partner ---> Müşterinin bir ortağı olup olmadığı(Evet, Hayır)
# Dependents ---> Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı(Evet, Hayır
# tenure ---> Müşterinin şirkette kaldığı ay sayısı
# PhoneService  --> Müşterinin telefon hizmeti olup olmadığı(Evet, Hayır),
# MultipleLines --> Müşterinin birden fazla hattı olup olmadığı(Evet, Hayır, Telefonhizmetiyok
# InternetService ---> Müşterinin internet servis sağlayıcısı(DSL, Fiber optik, Hayır)
# OnlineSecurity --_> Müşterinin çevrimiçi güvenliğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
# OnlineBackup--> Müşterinin online yedeğinin olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
# DeviceProtection ---> Müşterinin cihaz korumasına sahip olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
# TechSupport ---> Müşterinin teknik destek alıp almadığı(Evet, Hayır, İnternet hizmetiyok)
# StreamingTV ---> MüşterininTV yayını olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
# StreamingMovies--> Müşterinin film akışı olup olmadığı(Evet, Hayır, İnternet hizmetiyok)
# Contract---->Müşterinin sözleşme süresi(Aydan aya, Bir yıl, İkiyıl)
# PaperlessBilling---->Müşterinin kağıtsız faturası olup olmadığı(Evet, Hayır)
# PaymentMethod ---> Müşterinin ödeme yöntemi(Elektronikçek, Posta çeki, Banka havalesi(otomatik), Kredikartı(otomatik))
# MonthlyCharges ---> Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges ---> Müşteriden tahsil edilen toplam tutar
# Churn ----> Müşterinin kullanıp kullanmadığı(Evet veya Hayır)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
pd.set_option("display.width", 500)


def load():
    data = pd.read_csv("Dataset/Telco-Customer-Churn.csv")
    return data
def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    print("##########################################")
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                       "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count()}), end="\n\n")
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")
def outlier_thresholds(dataframe, col_name, q1=0.1, q3=0.9):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    # Nmerik görünülü kategorikleri çıkarttık
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
def check_df(dataframe, head=10):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    # Grafikler birbirini ezmesin diye
    plt.show(block=True)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


###############################################
# Görev 1: Keşifci Veri Analizi
###############################################

df = load()
df.head()

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız
df.info()
cat_cols, num_cols, cat_but_car = grab_col_names(df)


# Adım 2: Gerekli düzenlemeleri yapılmıştır. (Tip hatası olan değişkenler gibi)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip())

# Adım 3:  Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    cat_summary(df, col)

num_summary(df, num_cols)

for col in df.columns:
    plot_numerical_col(df, col)

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapılmıştır.

for col in num_cols:
    target_summary_with_num(df, "Churn", col)


cat_cols = [col for col in cat_cols if "Churn" not in col]
df[cat_cols]
df["_Churn"] = np.where(df["Churn"] == "Yes", 1, 0)
for col in cat_cols:
    target_summary_with_cat(df, "_Churn", col)

# Adım 5: Aykırı gözlem var mı inceleyiniz.

for i in num_cols:
    print(i,":",check_outlier(df,i))

# Adım 6: Eksik gözlem incelemesi
df.shape
missing_values_table(df)

###########################################################################
# Görev 2 : Feature Engineering
###########################################################################

# Adım 1:  Eksik ve aykırı gözlemler için gerekli işlemleri yapılmıştır.

# TotalCharges değişkeni içerisindeki boş değerlerin, tenure(Müşterinin şirkette kaldığı ay sayısı) incelendiğinde
# bu müşterilerin yeni müşteriler olduğu ve TotalCharges değişkenlerinin bu nedenle NAN olduğu anlaşılmaktadır.
# Veri setinden TotalCharges değişkenini silerek yanlış veya yanıltıcı sonuçlar elde edilebiliriz bundan dolayı
# boş değişkenleri 0 sabit değeri ile doldurmak veri seti için doğru bir karar olacaktır.

df["TotalCharges"].fillna(0, inplace=True)

# aykırı gözlem bulunmamaktadır

# Adım 2: Yeni değişkenler oluşturuldu.

df.corr()
df.corrwith(df["_Churn"]).sort_values(ascending=False)


# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# StreamingTV ve StreamingMovies Değişkenlerinin her ikisindede yararlanan müşteriler
df.loc[(df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes"), ["Streaming"]] = "Yes"
df.loc[~((df["StreamingTV"] == "Yes") & (df["StreamingMovies"] == "Yes")), ["Streaming"]] = "No"

# Her iki online hizmetten yararlanan müşteriler
df.loc[(df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes"), ["Online"]] = "Yes"
df.loc[~((df["OnlineSecurity"] == "Yes") & (df["OnlineBackup"] == "Yes")), ["Online"]] = "No"

# Herhangi bir destek, yedek veya koruma almayan kişiler
df["NEW_noProt"] = df.apply(lambda x: 1 if (x["OnlineBackup"] != "No") or (x["DeviceProtection"] != "No") or (x["TechSupport"] != "No") else 0, axis=1)

# Kişi otomatik ödeme yapıyor mu?
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Ortalama aylık ödeme
df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] +0.1)


# Adım 3:  Encoding işlemlerini gerçekleştirildi.

df.drop("_Churn",inplace=True, axis=1)

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
df = one_hot_encoder(df, ohe_cols)

df.head()


# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

num_cols
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

#########################################################
#Görev 3 : Modelleme
##########################################################

# Adım 1:  Sınıflandırma algoritmaları ile modeller kurulup, accuracy skorlarını indelendi ve en iyi 4 model seçildi.

y = df["Churn"]
X = df.drop(["Churn", "customerID"], axis=1)

################################################
# Modeling using CART
################################################

cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)

cv_results = cross_validate(cart_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7259701230724562
cv_results['test_f1'].mean()
# 0.4850566245690704
cv_results['test_roc_auc'].mean()
#  0.65015027220109

################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
# n_estimator --> birbirinden bağımsız fit edilecek ağaç sayısı
rf_model.get_params()

cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7948293439716311
cv_results['test_f1'].mean()
# 0.558036093388181
cv_results['test_roc_auc'].mean()
# 0.8268735708864314


################################################
# GBM (Gradient Boosting Machines)
################################################

gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.8032091788179881
cv_results['test_f1'].mean()
# 0.5852399606278981
cv_results['test_roc_auc'].mean()
# 0.846091221725381

################################################
# XGBoost
################################################
xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
xgboost_model.get_params()
cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
# 0.7850358289244467
cv_results['test_f1'].mean()
# 0.5601273834074249
cv_results['test_roc_auc'].mean()
# 0.8256916611690818

################################################
# LightGBM
################################################
lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])


cv_results['test_accuracy'].mean()
#   0.7931294962578231
cv_results['test_f1'].mean()
# 0.5710171184077059
cv_results['test_roc_auc'].mean()
# 0.8340725108215586

################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.7998017009161883
cv_results['test_f1'].mean()
# 0.5766472918512695
cv_results['test_roc_auc'].mean()
# 0.841432950939442


# En iyi 4 Model --->
# 1-)GBM
# 2-) LGBM
# 3-) CatBoost
# 4- ) Random Forests

# Adım 2: Hiperparametre Optimizasyonu

################################################
# GBM (Gradient Boosting Machines)
################################################
gbm_model = GradientBoostingClassifier(random_state=17)
gbm_model.get_params()
gbm_params = {"learning_rate": [0.01, 0.1],
              "max_depth": [3, 8, 10],
              "n_estimators": [100, 500,800, 1000],
              "subsample": [1, 0.5, 0.7, 0.1]}

gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

gbm_best_grid.best_params_

gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)


cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8050550640363895
cv_results['test_f1'].mean()
#0.5866170521623755
cv_results['test_roc_auc'].mean()
#0.8478843221973982

################################################
# LightGBM
################################################

lgbm_model = LGBMClassifier(random_state=17)
lgbm_model.get_params()

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

lgbm_best_grid.best_params_

lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
#0.8033524340280019
cv_results['test_f1'].mean()
# 0.588410301901684
cv_results['test_roc_auc'].mean()
# 0.8427305924104296


################################################
# CatBoost
################################################

catboost_model = CatBoostClassifier(random_state=17, verbose=False)

catboost_params = {"iterations": [200, 300, 500, 800],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}

catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
catboost_best_grid.best_params_
catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8064754177688883
cv_results['test_f1'].mean()
# 0.5849501148159314
cv_results['test_roc_auc'].mean()
# 0.8486261015451936


################################################
# Random Forests
################################################

rf_model = RandomForestClassifier(random_state=17)
# n_estimator --> birbirinden bağımsız fit edilecek ağaç sayısı
rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

rf_best_grid.best_params_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results['test_accuracy'].mean()
#0.8006503868471955
cv_results['test_f1'].mean()
#0.57102391461968
cv_results['test_roc_auc'].mean()
# 0.8458646289236397




# Adım 3:  Modele en çok etki eden değişkenleri gösteriniz ve önem sırasına
# göre kendi belirlediğiniz kriterlerde değişken seçimi yapıp seçtiğiniz
# değişkenler ile modeli tekrar çalıştırıp bir önceki model skoru arasındaki farkı gözlemleyiniz.


################################################
# LightGBM
################################################


cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# 0.8042039002516292
cv_results['test_f1'].mean()
# 0.5754627698131165
cv_results['test_roc_auc'].mean()
# 0.8461635106461355


def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm_final, X)
# önem düzeylerini görememizi sağlar


y_new = df["Churn"]
X_new = df[["MonthlyCharges","TotalCharges","tenure",
        "PaperlessBilling","gender","SeniorCitizen","Streaming","PhoneService"]]

cv_results_new = cross_validate(lgbm_final, X_new, y_new, cv=5, scoring=["accuracy", "f1", "roc_auc"])

cv_results_new['test_accuracy'].mean()
#0.7917096465901026
cv_results_new['test_f1'].mean()
# 0.5403333731578315
cv_results_new['test_roc_auc'].mean()
# 0.822864628297550


# Bonus: Hedef değişkenin dengesiz dağılımını gidermek için neler yapılmalı.
# Bu işlemleri yaptıktan sonra modeli tekrar kuruduğunuzda skorda bir farklılık oluştu mu?
# İnceleyiniz


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,f1_score,recall_score,roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Hedef değişkenin veri içerisinde görselleştirilmesi
f,ax=plt.subplots(1,2,figsize=(18,8))
df['Churn'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('dağılım')
ax[0].set_ylabel('')
sns.countplot('Churn',data=df,ax=ax[1])
ax[1].set_title('Churn')
plt.show()

# Hedef değişkenin dengesiz dağılımını gidermek için
# Resampling
# Daha fazla veri toplanabilir
# Sınıflandırma modellerinde bulunan “class_weight” parametresi kullanılarak azınlık ve çoğunluk sınıflarından eşit şekilde öğrenebilen model yaratılması,

########################################
# Resampling
########################################


# Yeniden örnekleme(Resampling), azınlık sınıfına yeni örnekler ekleyerek
# veya çoğunluk sınıfından örnekler  çıkarılarak veri setinin daha dengeli hale getirilmesidir.

# Oversampling
#Random Oversampling

# Azınlık sınıfından rastgele seçilen örneklerin eklenmesiyle veri setinin dengelenmesidir.
# Veri setiniz küçükse bu teknik kullanılabilinir.
# Overfitting’e neden olabilir.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)

# random oversampling önce eğitim setindeki sınıf sayısı
y_train.value_counts()


# RandomOver Sampling uygulanması (Eğitim setine uygulanıyor)
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
X_randomover, y_randomover = oversample.fit_resample(X_train, y_train)
y_randomover.value_counts()

lgbm_model = LGBMClassifier(random_state=17)
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}
lgbm_best_grid_new = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_randomover, y_randomover)
lgbm_final_new = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X_randomover, y_randomover)

y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)


#SMOTE Oversampling

from imblearn.over_sampling import SMOTE
oversample = SMOTE()
X_smote, y_smote = oversample.fit_resample(X_train, y_train)


lgbm_model = LGBMClassifier(random_state=17)
lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_best_grid_new = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X_smote, y_smote)
lgbm_final_new = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X_smote, y_smote)


y_pred = cart_model.predict(X_test)
y_prob = cart_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
roc_auc_score(y_test, y_prob)
