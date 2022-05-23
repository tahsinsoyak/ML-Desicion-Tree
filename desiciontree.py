import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder




#train verimizi aldık
train = pd.read_csv('titanicdata.csv')
#print(train.head())
import warnings
warnings.simplefilter("ignore")

#sayisal değerler ile kategorik değerleri ayırıyoruz
train_num = train[['Age', 'SibSp', 'Parch', 'Fare']]
train_cat = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]


#verilerin figür olarak gösterimi için
fig1, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows=2, ncols=2)

ax0.hist(train_num['Age'],bins=10,rwidth=5,color='red')
ax0.set_title('Age',loc = 'left', fontsize = 10)
ax0.grid(True)
ax0.set_xlabel('Yaş Aralığı',fontsize=8)
ax0.set_ylabel('Kişi Sayısı',fontsize=8)
#30 yaşında 15 kişi

ax1.hist(train_num['SibSp'],bins=10,rwidth=5,color='blue')
ax1.set_title('Sibling/Spouse',loc = 'left', fontsize = 10)
ax1.grid(True)
ax1.set_xlabel('Kardeş/Eş Sayısı',fontsize=8)
ax1.set_ylabel('Kişi Sayısı',fontsize=8)
#5 kardeşli 3 kişi

ax2.hist(train_num['Parch'],bins=10,rwidth=1,color='green')
ax2.set_title('Parent/Child',loc = 'left', fontsize = 10)
ax2.grid(True)
ax2.set_xlabel('Ebeveyn/Çocuk Sayısı',fontsize=8)
ax2.set_ylabel('Kişi Sayısı',fontsize=8)
#1 çocuklu 5 kişi


ax3.hist(train_num['Fare'],bins=10,rwidth=5,color='yellow')
ax3.set_title('Fare',loc = 'left', fontsize = 10)
ax3.grid(True)
ax3.set_xlabel('Bilete Ödenen Ücret',fontsize=8)
ax3.set_ylabel('Kişi Sayısı',fontsize=8)
#500 ödeyen 10 kişi


fig1.suptitle('Verimizdeki Sayısal Değerlerin Histogram Gösterimi')
fig1.tight_layout()
plt.show()



Survived = train_cat["Survived"].value_counts(normalize=True)
Survived = pd.DataFrame(Survived)
#yaşayan ve yaşamayanların oranını gösteriyoruz
#print(Survived)
''' 0 öldü 1 yaşadı
Survived
0	0.616162
1	0.383838
'''



Survived = Survived.reset_index()
#kolonları değiştirioyruz
Survived = Survived.rename(columns={"index":"Survived", "Survived":"% of Survivors"})
#yüzdelik dilime dönüştürüyoruz
Survived["% of Survivors"]= Survived["% of Survivors"]*100
#yuvarlıyoruz
Survived["% of Survivors"]= np.round(Survived["% of Survivors"],2)
'''
    Survived	% of Survivors
0	0	        61.62
1	1	        38.38
'''


Pclass = train_cat["Pclass"].value_counts(normalize=True)
Pclass = pd.DataFrame(Pclass)
Pclass = Pclass.reset_index()
Pclass = Pclass.rename(columns={"index":"Class", "Pclass":"% of passengers"})
Pclass = Pclass.sort_values(by='Class')
Pclass["% of passengers"]= Pclass["% of passengers"]*100
Pclass["% of passengers"]= np.round(Pclass["% of passengers"],2)
'''
	Class	% of passengers
1	1	    24.24
2	2	    20.65
0	3	    55.11
'''

Sex = train_cat["Sex"].value_counts(normalize=True)
Sex = pd.DataFrame(Sex)
Sex = Sex.reset_index()
Sex = Sex.rename(columns={"index":"Sex", "Sex":"% of passengers"})
Sex = Sex.sort_values(by='Sex')
Sex["% of passengers"]= Sex["% of passengers"]*100
Sex["% of passengers"]= np.round(Sex["% of passengers"],2)


Ticket = train_cat["Ticket"].value_counts(normalize=True)
Ticket = pd.DataFrame(Ticket)
Ticket = Ticket.reset_index()
Ticket = Ticket.rename(columns={"index":"Ticket", "Ticket":"% of passengers"})
Ticket["% of passengers"]= Ticket["% of passengers"]*100
Ticket["% of passengers"]= np.round(Ticket["% of passengers"],2)

Cabin = train_cat["Cabin"].value_counts(normalize=True)
Cabin = pd.DataFrame(Cabin)
Cabin = Cabin.reset_index()
Cabin = Cabin.rename(columns={"index":"Cabin", "Cabin":"% of passengers"})
Cabin["% of passengers"]= Cabin["% of passengers"]*100
Cabin["% of passengers"]= np.round(Cabin["% of passengers"],2)


Embarked = train_cat["Embarked"].value_counts(normalize=True)
Embarked = pd.DataFrame(Embarked)
Embarked = Embarked.reset_index()
Embarked = Embarked.rename(columns={"index":"Embarked", "Embarked":"% of passengers"})
Embarked["% of passengers"]= Embarked["% of passengers"]*100
Embarked["% of passengers"]= np.round(Embarked["% of passengers"],2)


sns.set(font_scale=0.8)
fig2, axes = plt.subplots(2, 3,  sharey=True)
fig2.suptitle('Verimizdeki Kategorik Değerlerin Barplot Gösterimi')
sns.barplot(ax=axes[0,0], x='Survived', y='% of Survivors', data=Survived)
axes[0,0].set_title('Survive Rate')

sns.barplot(ax=axes[0,1], x='Class', y='% of passengers', data=Pclass)
axes[0,1].set_title('Percent of Pclass')

sns.barplot(ax=axes[0,2], x='Sex', y='% of passengers', data=Sex)
axes[0,2].set_title('Percent of Gender')


'''
sns.barplot(ax=axes[1,0], x='Ticket', y='% of passengers', data= Ticket)
axes[1,0].set_title('Percent of Ticket')

sns.barplot(ax=axes[1,1], x='Cabin', y='% of passengers', data= Cabin)
axes[1,1].set_title("Percent of Cabin")
'''

sns.barplot(ax=axes[1,2], x='Embarked', y='% of passengers', data= Embarked)
axes[1,2].set_title("Percent of Embarkation")

#tüm barplot gösterimimiz bitmiştir
plt.show()

data = pd.read_csv('titanicdata.csv')
#print(data.columns)

#ihtiyacımız olmayan kolonları siliyoruz.
data = data[['Survived','Pclass','Sex','Age','Fare']]

#print(data)
#cinsiyet için yaş ortalamasını alıyoruz.
data.groupby('Sex')['Age'].mean()

#data.isnull().sum()

#cinsiyete göre boş kolonları dolduruyoruz.
data['Age']=data.groupby("Sex")['Age'].transform(lambda x: x.fillna(x.mean()))
#lambda expression kullanarak ortalamalı dolduruyoruz.


'''
# import category encoders
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Sex'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)
#1 ve 2 olarak güncelledi.
'''


#label encoder kullanalım onun yerine
le = LabelEncoder()
data['Sex'] = le.fit_transform(data.Sex)
#cinsiyet 1 male ve 0 female olarak ayarlandı

from sklearn.model_selection import train_test_split

X = data.drop(['Survived'], axis=1)
y = data['Survived']


#train test split fonksiyonu verileri rastgele bölüp sub arrayler yaratıyor. [test ve train için]
#test %30 belirliyoruz. karıştırma için rastgelelik 8  veriyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#X elimizdeki veriler [programa verdik] %70 i 623 veri %30 u 268 veri
#y hedefimizi [programa vedik]

#GİNİ KULLANARAK YAPALIM

from sklearn.tree import DecisionTreeClassifier
#gini indeksi ile DecisionTreeClassifier modeli oluşturuyoruz.
#max depth karar ağacındaki derinlik sayımız
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=10,min_samples_leaf=8)
#gini classifier fitliyoruz train ile
clf_gini.fit(X_train, y_train)
#predict ediyoruz
y_pred_gini = clf_gini.predict(X_test)

#x test verilerimizle tahmin yaptıp
#y test verilerimize tahmin ettiriyorruz
from sklearn.metrics import accuracy_score
print('Gini ile modelin doğruluğu: {0:0.8f}'. format(accuracy_score(y_test, y_pred_gini)))
y_pred_train_gini = clf_gini.predict(X_train)
print('Train set ile doğruluk oranı: {0:0.8f}'. format(accuracy_score(y_train, y_pred_train_gini)))


from sklearn import tree
fig3 = plt.figure(figsize=(12,8))
features = X_train.columns
tree.plot_tree(clf_gini.fit(X_train, y_train),feature_names=features,filled=True)
fig3.suptitle('Gini Karar Agacı', fontsize=20)
plt.show()

#çizilen ağaçta
'''
623 veriden erkek ve kadın olanlarını ayırıyor 
kadın olanlar sola erkek olanlar sağa.
396 erkek 227 kadın

623 taneden 381 ölü 242 yaşayan


227 kadından Pclass 2nin aldında olanlar sola 127 kişi
2nin üstü sağa 100 kişi

227 taneden 57 ölü 170 yaşayan

127 taneden age 2.5 altında solanlar sola 2 kişi
üstünde olanlar sağa 125 kişi

6 ölü 121 yaşayan

'''


#Entropi ile hesaplama

#leaf  Bir yaprak düğümde olması gereken minimum numune sayısı.
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=10,min_samples_leaf=8)
# modeli fitliyelim
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)
print('Entropi ile modelin doğruluğu: {0:0.8f}'. format(accuracy_score(y_test, y_pred_en)))
y_pred_train_en = clf_en.predict(X_train)
y_pred_train_en
print('Train set ile doğruluk oranı: {0:0.8}'. format(accuracy_score(y_train, y_pred_train_en)))

fig3 = plt.figure(figsize=(12,8))
tree.plot_tree(clf_en.fit(X_train, y_train),feature_names=features,filled=True) 
fig3.suptitle('Entropy Karar Agacı', fontsize=20)
plt.show()

#derinliği ve yaprak bakma sayısını aynı kullandığımızdan 




#hata matrisi olarak da bilinen bir karışıklık matrisi,
#  tipik olarak denetimli bir öğrenme algoritması olan bir
#  algoritmanın performansının görselleştirilmesine izin veren
#  özel bir tablo düzenidir.

from sklearn.metrics import confusion_matrix,plot_confusion_matrix

#prediction ile observe yani yest kısmındaki bağıntılara bakıyor
cf_matrix = confusion_matrix(y_test, y_pred_en)
fig4 = plt.figure(figsize=(4,4))
sns.heatmap(cf_matrix,annot=True)
fig4.suptitle('Confusion Matrix Isı Haritası', fontsize=10)
#[[146  22]
# [ 28  72]]
plt.show()


'''
fig5 = plt.figure(figsize=(4,4))
sns.set_theme(style="darkgrid")
sns.boxplot(x= "Survived", y="Age", data = data)
fig5.suptitle('Yaşam/Ölüm Age dağılımı', fontsize=10)
#plt.show()


aykırı veriler 


üst aşırı
üstçeyrek
medyan
altçeyrek
bıyık
alt aşırı
'''

#AĞAÇ BUDAMA

#1. Pre pruning techniques


'''
Zaten bu teknikleri kullanarak budama yaptık

max_depth: karar ağacının maksimum derinliği
min_sample_split: Bir dahili düğümü bölmek için gereken minimum örnek sayısı
min_samples_leaf: Bir yaprak düğümde olması gereken minimum örnek sayısı.

'''


#2. Post pruning techniques

'''
Maliyet karmaşıklığı budama
Alfa için doğru parametreyi bulmakla ilgilidir.
Bu ağaç için alfa değerlerini alacağız
ve budanan ağaçlarla doğruluğunu kontrol edeceğiz.
'''
clf = tree.DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
#print(ccp_alphas)


# Her alfa için modelimizi bir listeye ekleyeceğiz
clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)


#clfs ve alphas içindeki son elemanı çıkarıyoruz. çünkü 1 düğümlü ağaç.
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]
node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig6 = plt.figure(figsize=(4,4))
plt.scatter(ccp_alphas,node_counts)
plt.scatter(ccp_alphas,depth)
plt.plot(ccp_alphas,node_counts,label='dügüm sayısı',drawstyle="steps-post")
plt.plot(ccp_alphas,depth,label='derinlik',drawstyle="steps-post")
fig6.suptitle('Alpha değerleri dağılımı', fontsize=10)
plt.legend()
plt.show()
#Alfa arttıkça düğüm sayısı ve derinlik azalır



#her biri için accuracy bakalım
train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(X_train)
    y_test_pred = c.predict(X_test)
    train_acc.append(accuracy_score(y_train_pred,y_train))
    test_acc.append(accuracy_score(y_test_pred,y_test))

plt.scatter(ccp_alphas,train_acc)
plt.scatter(ccp_alphas,test_acc)
plt.plot(ccp_alphas,train_acc,label='train_accuracy',drawstyle="steps-post")
plt.plot(ccp_alphas,test_acc,label='test_accuracy',drawstyle="steps-post")
plt.legend()
plt.title('Accuracy vs alpha',fontsize=8)
plt.show()
#doğruluk ile alpha değerlerinin çizimi

#0.002 alabiliriz alphayı


clf_ = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.002)
clf_.fit(X_train,y_train)
y_train_pred = clf_.predict(X_train)
y_test_pred = clf_.predict(X_test)

print('Train score: {0:0.8}'. format(accuracy_score(y_train_pred,y_train)))
print('Test score: {0:0.8}'. format(accuracy_score(y_test_pred,y_test)))
plt.figure(figsize=(12,8))
tree.plot_tree(clf_,feature_names=features,filled=True)
plt.title('Alfa Değeri Verilmiş Ağac Sınıflandırması',fontsize=18)
plt.show()



#Raspberry için

'''
lcd kütüphanesi komut sisteminden yüklenir.
git clone https://github.com/the-raspberry-pi-guy/lcd


import lcddriver # lcd ekranı kullanabilemek için kurduğumuz kütüphaneyi ekledik.
import time      # zamanlama kullanacağımız için time kütüphanesini ekledik.
display = lcddriver.lcd()
try:
    while True:
        display.lcd_display_string('Train score: {0:0.8}'. format(accuracy_score(y_train_pred,y_train)), 1) # 1. satır için ekranda görünmesini istediğimiz metni yazdık
        display.lcd_display_string('Test score: {0:0.8}'. format(accuracy_score(y_test_pred,y_test)), 2) # 2. satır için yazdırmak istediğimiz metni yazdırdık.                                 
except KeyboardInterrupt: # programdan çıkana kadar çalışmasını sağladık.
    display.lcd_clear()

'''