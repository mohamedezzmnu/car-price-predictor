# CarPulse — Vehicle Valuation Engine (Liquid Glass UI)

تطبيق ويب لتقدير القيمة السوقية للسيارات بتصميم **Liquid Glass** مع واجهة مبسطة تركّز على المواصفات الأساسية وخيارات متقدمة قابلة للطي.

## هيكل المشروع

```text
car-price-predictor/
├── index.html                         # الواجهة الرئيسية
├── styles.css                         # نظام تصميم Liquid Glass المعماري
├── app.js                             # محرك التنبؤ الرياضي والتفاعلات
├── model.json                         # معاملات النموذج المدرب
├── vercel.json                        # إعدادات الاستضافة على Vercel
├── assets/
│   └── model_diagnostics.png          # رسومات فحص النموذج
└── ml/
    ├── car_price_linear_regression.ipynb
    └── car_price_linear_regression.py
```
