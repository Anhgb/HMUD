import pandas as pd
from sklearn.preprocessing import LabelEncoder
# dl mẫu 
data= {
    'car': ['Toyota', 'Honda', 'Nissan', 'Toyota', 'Honda'],
    'country': ['Japan', 'Japan', 'Japan', 'Japan', 'Japan'],
    'label': ['Sendan', 'SUV', 'Sendan', 'Hatchback', 'SUV']
}
df = pd.DataFrame(data)
print("\n Dữ liệu gốc:")
print(df)
# KHởi tạo đối tượng LabelEncoder
label_encoder = LabelEncoder()
# Fit và transform cột 'label'
# .fit_transform() sẽ học các nhãn đán duy nhất và mã hóa chúng 
df['label_encoded'] = label_encoder.fit_transform(df['label'])
print("\n DataFrame sau khi mã hóa nhãn")
print(df)
# Chúng ta có thể xen ánh xạ giũa các nhãn gốc và nhãn đã mã hóa 
# .classes_ thuộc tính này chứa các nhãn gốc thứ tự đã đc mã hóa
print("\n --- Anh xạ nhãn ---")
for  i, label in enumerate(label_encoder.classes_):
    print(f"Nhãn '{label}' được mã hóa thành {i}")
# Hoặc dùng dict(zip(...)) để dễ nhìn hơn
mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\nMapping dạng từ điển:", mapping)