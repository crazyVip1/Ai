import kagglehub
import shutil
import os

print("جاري تنزيل قاعدة البيانات من Kaggle...")
# 1. تنزيل البيانات من Kaggle
download_path = kagglehub.dataset_download("shanegerami/ai-vs-human-text")

# 2. نسخ جميع الملفات إلى مجلد مشروعك الحالي
current_dir = os.getcwd()

for file_name in os.listdir(download_path):
    full_file_name = os.path.join(download_path, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, current_dir)
        print(f"تم نقل الملف بنجاح إلى مشروعك: {file_name}")

print("\nاكتملت العملية! افحص قائمة الملفات على اليسار.")
