# AutoValuate AI — Car Price Prediction (Liquid Glass Edition)

واجهة ويب تفاعلية احترافية بتصميم **Liquid Glass (Apple VisionOS / Translucent Refractive UI)** مبنية لتشغيل نموذج **Linear Regression** المدرب على بيانات أسعار السيارات (`car_price_prediction.csv`).

## هيكل المشروع (Project Structure)

```text
car-price-predictor/
├── index.html                         # الواجهة الرئيسية بتصميم Liquid Glass وتفاعلات لحظية
├── styles.css                         # نظام التصميم والـ Liquid Glass Shaders & Specular Borders
├── app.js                             # محرك التنبؤ الرياضي المباشر (Linear Regression Inference Engine)
├── model.json                         # معاملات الموديل المدرب (Intercept, Coefficients, Scaler Mean/Scale)
├── vercel.json                        # إعدادات النشر الفوري على Vercel
├── README.md                          # دليل المشروع وخطوات الرفع
├── assets/
│   └── model_diagnostics.png          # رسومات فحص الموديل (Actual vs Predicted & Residuals)
└── ml/
    ├── car_price_linear_regression.ipynb  # نوتبوك التدريب التفاعلي (Colab / Jupyter)
    └── car_price_linear_regression.py     # سكريبت البايثون الأصلي لتدريب الموديل وتوليد model.json
```

## المميزات التقنية

1. **Liquid Glass UI Architecture**:
   - طبقات زجاجية شفافة متعددة العمق (`backdrop-filter: blur(32px) saturate(190%)`) مع انكسار ضوئي SVG (`feTurbulence` + `feDisplacementMap`).
   - إضاءة حواف انعكاسية (`Specular Rim Highlights`) تتبع مؤشر الماوس بنعومة.
   - خلفيات ضوئية عضوية متحركة (Ambient Mesh Orbs) تمنح عمقاً بصرياً مريحاً للعين.
2. **محرك استدلال رياضي مطابق 100% لكود البايثون**:
   - حساب عمر السيارة (`Car age = 2020 - Prod. year`)
   - التحويل اللوغاريتمي للمسافة المقطوعة (`Math.log1p(Mileage)`)
   - التوحيد القياسي (`StandardScaler`: `(x - mean) / scale`) للأعمدة الرقمية السبعة
   - ترميز الفئات (`One-Hot Encoding` مع `drop_first=True`) لـ 61 شركة تصنيع، 11 فئة هيكل، 7 أنواع وقود، 4 نواقل حركة، و3 أنظمة دفع
   - حساب التوقع اللوغاريتمي ثم العودة للسعر الحقيقي عبر `Math.expm1(y_log)` مع نطاق الثقة الإحصائي (`±1σ`).
3. **لوحة تحليلات تفاعلية (Explainable AI & Diagnostics)**:
   - عرض تفصيلي لكيفية مساهمة كل ميزة في رفع أو خفض السعر.
   - قوالب سيارات جاهزة للتجربة بنقرة واحدة + أداة مقارنة بين سيارتين.
   - عرض مقاييس دقة النموذج (`R² = 0.525`, `MAE = $6,878`, `N = 13,174`) ورسومات التشخيص.

## خطوات الرفع على GitHub ثم Vercel

1. فك الضغط عن مجلد `car-price-predictor`.
2. أنشئ مستودع (Repository) جديد على **GitHub** وارفع بداخله جميع ملفات المجلد.
3. ادخل على **Vercel** واضغط **Add New Project** واختر المستودع من GitHub.
4. اترك إعدادات البناء كما هي (`Framework Preset: Other`) واضغط **Deploy**.
