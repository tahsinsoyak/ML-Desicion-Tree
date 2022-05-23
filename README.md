# Makine Öğrenmesi İle Titanik Gemisinden Hayatta Kalma Tahmini


### 1.Kütühpaneler;
•	Pandas: Bu kütüphane temel olarak zaman etiketli serileri ve sayısal tabloları işlemek için bir veri yapısı oluşturur ve bu şekilde çeşitli işlemler bu veri yapısı üzerinde gerçekleştirilebilir.
•	Numpy: Çok boyutlu dizileri ve matrisleri destekleyen, bu diziler üzerinde çalışacak üst düzey matematiksel işlevler ekleyen bir kitaplıktır.
•	Sklearn: Makine öğrenimi kitaplığıdır.
•	Matplotlib: Sayısal matematik uzantısı NumPy için bir çizim kitaplığıdır. Tkinter, wxPython, Qt veya GTK gibi genel amaçlı GUI araç takımlarını kullanarak grafikleri uygulamalara gömmek için nesne yönelimli bir API sağlar.
•	Seborn: Seaborn, matplotlib tabanlı bir Python veri görselleştirme kütüphanesidir.



### 2.Veri Analizi;
Kütüphaneler tanımladıktan sonra veriler hakkında bilgi edinmek amacı ile verilerin grafiklerini çizdirilmelidir. Grafikleri çizdirirken eğer veriler farklı türlerde ise numerik ve kategorik veriler farklı şekilde matplot yardımıyla gösterilmeli.
Numerik verilerimiz bir figürde toplanıp yan yana histogram aracılığıyla gösterildi.

![image](https://user-images.githubusercontent.com/54424377/169784859-639e98b2-28c0-4a8e-938d-b68e5e7ccfb4.png)
##### Şekil 1'de görüldüğü gibi histogram şeklinde verilerimiz dağılımı gösterilmiştir.



Örneğin; Yaş verisi 20-40 aralığında olanların daha fazla olduğunu gösteriyor. Kardeş eş sayısınında hiç kardeşi eşi olmayanların sayısının 600’ün üstünde olduğunu görürürüz.

Genelde kategorik verilerin gösterimini ve analizini barplot yardımıyla yapılır.
![image](https://user-images.githubusercontent.com/54424377/169784953-8678cb95-96ce-449f-b312-3f7191464d53.png)
##### Şekil 2'de görüldüğü gibi barplot şeklinde verilerimiz dağılımı gösterilmiştir.

