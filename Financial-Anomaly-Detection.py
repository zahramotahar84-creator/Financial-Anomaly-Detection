import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler
df = pd.read_csv("C:\\Users\\user\\Downloads\\Credit_Card_Applications.csv")
# فرض می‌کنیم نام دیتاست شما df است
# ۱. حذف ستون غیرضروری
if 'Customer Id' in df.columns:
    df = df.drop(columns=['Customer Id'])
    # ۲. مدیریت داده‌های پرت (Outliers) با RobustScaler
    # این مقیاس‌بند بر اساس چارک‌ها عمل می‌کند و در برابر داده‌های پرت مقاوم است
robust_cols = ['A2', 'A3', 'A13', 'A14', 'A10']
scaler_robust = RobustScaler()
df[robust_cols] = scaler_robust.fit_transform(df[robust_cols])
# ۳. نرمال‌سازی ستون‌های عددی دیگر (اگر ستون عددی دیگری بجز دسته‌ای‌ها مانده باشد)
# در اینجا فعلاً روی بقیه ستون‌ها MinMaxScaler می‌زنیم
# (البته بعد از One-Hot، ستون‌های جدید خودبخود بین 0 و 1 خواهند بود)
# ۴. اعمال One-Hot Encoding برای ستون‌های دسته‌ای
# این کار ستون‌های A1, A4, A8, A9, A11, A12 رو به ستون‌های 0 و 1 تبدیل می‌کنه
categorical_cols = ['A1', 'A4', 'A8', 'A9', 'A11', 'A12']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# ۵. جدا کردن ویژگی‌ها (X) از هدف (y)
# ستون Class هدف ماست (تقلب یا سالم)
X = df.drop(columns=['Class'])
y = df['Class']
print("پیش‌پردازش با موفقیت انجام شد!")
print(f"تعداد ویژگی‌های نهایی بعد از One-Hot: {X.shape[1]}")
print(df.head())
counts = df['Class'].value_counts(normalize=True)
print("درصد تراکنش‌های هر کلاس:")
print(counts * 100)
# ذخیره درصد ناهنجاری برای استفاده در مدل
outlier_fraction = counts[1]
print(f"\nمقدار دقیق ناهنجاری (contamination) پیشنهادی:{outlier_fraction:.2f}")
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
# افزایش تعداد درخت‌ها و تنظیم دقیق‌تر میزان حساسیت
if_optimized = IsolationForest(n_estimators=300,
max_samples='auto',
contamination=0.44, # طبق درصدی که قبلاً به دست آوردی
random_state=42)
# آموزش و پیش‌بینی
if_optimized.fit(X)
y_pred_if_opt = if_optimized.predict(X)
y_pred_if_final = [1 if i == -1 else 0 for i in y_pred_if_opt]
print("--- نتیجه Isolation Forest بهینه‌شده ---")
print(classification_report(y, y_pred_if_final))
# ۵. چاپ گزارش جدید
print("--- گزارش عملکرد روش نیمه‌نظارتی ---")
print(classification_report(y, y_pred_if_final))
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
# در مدل‌های نظارتی بهتر است داده‌ها را به دو بخش آموزش و تست تقسیم کنیم
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# تعریف مدل XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
# آموزش مدل با داده‌های لیبل‌دار
xgb_model.fit(X_train, y_train)
# پیش‌بینی روی داده‌های تست (که مدل قبلاً ندیده)
y_pred_xgb = xgb_model.predict(X_test)
print("--- نتیجه خیره‌کننده XGBoost ---")
print(classification_report(y_test, y_pred_xgb))
def check_transaction_realtime(new_data_row):
    """
    new_data_row: یک سطر از داده جدید که باید دقیقاً مثل داده‌های آموزشی پیش‌پردازش شده باشد
    """
    # پیش‌بینی توسط مدل آموزش دیده
    prediction = xgb_model.predict(new_data_row)
    # تفسیر نتیجه
    if prediction[0] == -1:
        return "⚠️ هشدار: این تراکنش مشکوک (ناهنجار) است!"
    else:
        return "✅ تراکنش سالم به نظر می‌رسد."
    # تست مدل با یک داده تصادفی از خودِ دیتاست (مثلاً سطر دهم)
sample_transaction = X.iloc[[10]]
result = check_transaction_realtime(sample_transaction)
print(f"نتیجه بررسی آنی: {result}")
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
# رسم ماتریس آشفتگی
cm = confusion_matrix(y,y_pred_if_final )
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted (Model)')
plt.ylabel('Actual (Truth)')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ۱. پیش‌بینی برای تمام داده‌ها با مدل XGBoost
y_pred_all_xgb = xgb_model.predict(X)
# ۲. محاسبه ماتریس آشفتگی
cm_xgb = confusion_matrix(y, y_pred_all_xgb)
# ۳. نمایش ماتریس
print("--- ماتریس آشفتگی XGBoost (کل داده‌ها) ---")
print(cm_xgb)
# رسم گرافیکی برای مقایسه راحت‌تر
plt.figure(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_xgb, display_labels=['Normal', 'Fraud'])
disp.plot(cmap='YlGnBu') # رنگ زرد و آبی برای تنوع
plt.title('XGBoost Confusion Matrix')
plt.show()
# ایجاد یک دیتافریم برای مقایسه
results_df = pd.DataFrame({
'Actual': y,
'Predicted': y_pred_if_final
})
# تعریف وضعیت‌ها
def check_status(row):
    if row['Actual'] == 1 and row['Predicted'] == 1:
        return 'Correct Fraud Detection (TP)'
    elif row['Actual'] == 0 and row['Predicted'] == 0:
        return 'Correct Normal Detection (TN)'
    elif row['Actual'] == 0 and row['Predicted'] == 1:
        return 'False Alarm (FP)'
    else:
        return 'Missed Fraud (FN)'
results_df['Status'] = results_df.apply(check_status, axis=1)
        # حالا استفاده از value_counts برای دیدن تعداد دقیق هر کدوم
print("آمار دقیق عملکرد مدل:")
print(results_df['Status'].value_counts())
        # اگر بخوای لیست کسانی که واقعاً تقلب بودن و درست حدس زدی رو ببینی:
correct_frauds = results_df[results_df['Status'] == 'Correct Fraud Detection (TP)']
print(f"\nتعداد کل کسانی که مچشون رو گرفتیم: {len(correct_frauds)}")
