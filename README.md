# 🤖 AI vs. Human Text Detector

نظام خبير ومعالجة نصوص ذكي لتمييز النصوص المنشأة بواسطة الذكاء الاصطناعي (AI-Generated) عن النصوص المكتوبة بواسطة العنصر البشري (Human-Written).

🔗 **رابط التطبيق المباشر (Live Demo):** [AI Detector App](https://htvvjezg9s7shcur6lrd4w.streamlit.app/)

---

## 📌 نبذة عن المشروع (Overview)

يهدف هذا المشروع إلى توفير أداة تحليل نصوص تفاعلية وسريعة تعتمد على تقنيات تعلم الآلة (Machine Learning) والمعالجة الطبيعية للغات (NLP). يقوم النظام بتحليل هيكل النص المدخل، وحساب الاحتمالية الإحصائية للتمييز بين الكتابة البشرية والنصوص التوليدية، وعرض النتائج عبر رسومات بيانية تفاعلية.

---

## 🛠️ التقنيات والمكتبات المستخدمة (Tech Stack)

* **Python 3.10+** - لغة البرمجة الأساسية.
* **Streamlit** - بناء الواجهة التفاعلية وتصميم ويب مستجيب.
* **Scikit-Learn** - بناء وتدريب نموذج التصنيف.
* **Plotly** - إنشاء المخططات الإحصائية التفاعلية.
* **Joblib** - حفظ واسترجاع تسلسل نموذج التعلم المكتمل (`ai_detector_pipeline.pkl`).
* **Pandas & NumPy** - إعداد ومعالجة البيانات الحسابية.

---

## 📂 هيكلية المشروع (Repository Structure)

```text
.
├── app.py                     # تطبيق الويب الرئيسي (Streamlit Interface)
├── train_model.py             # كود تدريب النموذج ومعالجة البيانات
├── ai_detector_pipeline.pkl   # خط المعالجة والنموذج المدرب المحفوظ
├── requirements.txt           # مكتبات الاعتماد الخاصة بالنشر أونلاين
├── .gitignore                 # استبعاد البيانات الضخمة والملفات المؤقتة
└── README.md                  # التوثيق الشامل للمشروع

🚀 كيفية تشغيل المشروع محلياً (Local Setup)
 * استنساخ المستودع (Clone the repository):
   git clone [https://github.com/crazyVip1/Ai.git](https://github.com/crazyVip1/Ai.git)
cd Ai

 * تثبيت المكتبات المطلوبة:
   pip install -r requirements.txt

 * تشغيل التطبيق محلياً:
   streamlit run app.py

☁️ النشر والتطوير (Deployment)
تم نشر هذا التطبيق أونلاين باستخدام Streamlit Community Cloud ومربوط مباشرة بفرع main على GitHub لضمان التحديث التلقائي للتطبيق مع كل commit جديد.
👤 إعداد وتطوير
المهندس: حسن الهاتف (Hassan Al-Hatef)
التخصص: هندسة الميكاترونكس (Mechatronics Engineering)

