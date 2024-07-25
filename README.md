**data preprocessing için:**
1) null değerler kontrol edildi.
2) binary değer dönüşümü yapıldı.
3) continous değerlerin (total_bill, size) tip (label) ile olan korelasyonuna bakıldı.
4) haftanın günleri sayısal degere cevrildi.
5) cinsiyete göre tip oranına bakıldı. (aralık çok yakın ama aykırı değerler var)
6) bütün veriler sayısala çevrildikten sonra cosine similarity bakıldı.
7) size ve total_bill çok benzer olduğu için tek feature'a indirgendi.
8) cinsiyetteki oluşan aykırı değerler mean değere çevrildi. df tekrar düzenlendi.

**model oluşturma:**
1) model fit edildi, mse hesaplandı
2) sınıflandırma algoritmaları için tip sütunu kategorize edildi (float->int tür dönüşümü ve label olarak ayarlama)
3) train ve test aralıklarının belirlenmesi
4) öğrenme algoritmalarının uygulanması ve accuracy score'ların hesaplanması.
